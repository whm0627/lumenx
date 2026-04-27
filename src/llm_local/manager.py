"""ModelManager — singleton, state machine, lock, idle watcher."""
from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

from ..utils.gpu_lock import GPULock
from .config import LocalLLMConfig
from .runtime import LocalRuntime
from .vram import QuantMode, detect_vram_total_mb, estimate_model_size_b, pick_quant_for_model

GPU_LOCK_NAME = "llm"

logger = logging.getLogger(__name__)


class CancelledByUser(Exception):
    """Raised inside the loader thread when a cancel was requested."""


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"  # not currently distinguished from LOADING; reserved
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


RuntimeFactory = Callable[..., LocalRuntime]


def _default_runtime_factory(
    hf_id: str,
    quant: QuantMode,
    gguf_file: Optional[str] = None,
    n_ctx: int = 65536,
):
    """Dispatch to EmbeddedServerRuntime for GGUF repos (subprocess llama-server),
    else LocalRuntime (in-process transformers for safetensors).

    n_ctx is forwarded to the GGUF runtime; LocalRuntime uses its own default.
    """
    from .gguf_utils import is_gguf_repo, list_gguf_files, pick_default_gguf_file

    if is_gguf_repo(hf_id):
        from .runtime_server import EmbeddedServerRuntime
        if gguf_file is None:
            files = list_gguf_files(hf_id)
            gguf_file = pick_default_gguf_file(files)
            if gguf_file is None:
                raise RuntimeError(f"No .gguf files found in repo {hf_id}")
        return EmbeddedServerRuntime(hf_id=hf_id, gguf_file=gguf_file, n_ctx=n_ctx)

    return LocalRuntime(hf_id=hf_id, quant=quant)


class ModelManager:
    """Asyncio singleton managing one LocalRuntime at a time."""

    _instance: Optional["ModelManager"] = None

    # Quant step-down sequence used when an OOM is hit during load (spec §9).
    _OOM_STEP_DOWN = {
        QuantMode.FP16: QuantMode.INT8,
        QuantMode.BF16: QuantMode.INT8,
        QuantMode.INT8: QuantMode.INT4,
        QuantMode.INT4: None,
    }

    def __init__(
        self,
        config: LocalLLMConfig,
        runtime_factory: RuntimeFactory = _default_runtime_factory,
        watcher_interval_sec: float = 1.0,
    ):
        self.config = config
        self._runtime_factory = runtime_factory
        self._watcher_interval = watcher_interval_sec
        self._runtime: Optional[LocalRuntime] = None
        self._state = ModelState.UNLOADED
        self._error_msg: Optional[str] = None
        self._last_used_ts = time.time()
        self._lock = asyncio.Lock()
        self._watcher_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Thread id of the worker currently running runtime.load() (set inside
        # the executor wrapper, used by cancel() for ctypes async-raise).
        self._loader_thread_id: Optional[int] = None
        self._cancel_requested = False
        # Register with GPULock so the image runtime (or any other) can
        # evict us before claiming VRAM. Eviction calls unload_sync from
        # whichever thread the evictor runs on.
        GPULock.get().register(GPU_LOCK_NAME, self.unload_sync)

    # ---- singleton accessor used by LLMAdapter ----

    @classmethod
    def install(cls, config: LocalLLMConfig) -> "ModelManager":
        cls._instance = ModelManager(config=config)
        return cls._instance

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            raise RuntimeError("ModelManager not installed; call install() at startup.")
        return cls._instance

    # ---- public async API ----

    async def configure(self, new_config: LocalLLMConfig) -> None:
        async with self._lock:
            id_changed = new_config.hf_id != self.config.hf_id
            quant_changed = new_config.quant != self.config.quant
            self.config = new_config
            if (id_changed or quant_changed) and self._state == ModelState.READY:
                await self._unload_locked()

    async def cancel(self) -> bool:
        """Abort an in-flight load OR clear an ERROR state. Returns True if
        any state transition happened.

        - LOADING/DOWNLOADING: injects CancelledByUser into the loader thread
          via ctypes.PyThreadState_SetAsyncExc, waits for it to exit, then
          resets state. Best-effort during HF socket I/O.
        - ERROR: just resets state to UNLOADED + clears error message. Lets
          the user dismiss a stuck error and recover the cache via the same
          /cancel endpoint that handles file deletion.
        - UNLOADED/READY: no-op (returns False).

        File cleanup (deleting partial blobs / failed snapshots) is handled
        by the caller (/llm/local/cancel endpoint) so the manager stays
        purely about runtime state.
        """
        if self._state not in (
            ModelState.LOADING, ModelState.DOWNLOADING, ModelState.ERROR
        ):
            return False

        # ERROR state: no thread to interrupt, just reset and return.
        if self._state == ModelState.ERROR:
            logger.info("Cancel: clearing ERROR state")
            async with self._lock:
                if self._runtime is not None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
                    except Exception:
                        pass
                    self._runtime = None
                self._state = ModelState.UNLOADED
                self._error_msg = None
            return True

        # LOADING / DOWNLOADING: interrupt the worker thread.
        self._cancel_requested = True
        logger.info("Cancel requested for in-flight load")

        if self._loader_thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._loader_thread_id),
                ctypes.py_object(CancelledByUser),
            )
            if res != 1:
                logger.warning(f"PyThreadState_SetAsyncExc returned {res}; thread may have already exited")

        # Wait up to ~4 seconds for the thread to notice and exit
        for _ in range(20):
            if self._state != ModelState.LOADING:
                break
            await asyncio.sleep(0.2)

        # Force-unload + reset state. Take the lock so we don't race with
        # an ongoing chat() (impossible during LOADING, but defensive).
        async with self._lock:
            if self._runtime is not None:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
                except Exception:
                    pass
                self._runtime = None
            self._state = ModelState.UNLOADED
            self._error_msg = None

        self._cancel_requested = False
        return True

    async def load(self) -> Dict:
        async with self._lock:
            await self._load_locked()
            return self.status()

    async def unload(self) -> None:
        async with self._lock:
            await self._unload_locked()

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        async with self._lock:
            if not self.config.hf_id:
                raise RuntimeError(
                    "Configure local LLM first (set HF model id in Settings)."
                )
            if self._state != ModelState.READY:
                await self._load_locked()
            assert self._runtime is not None
            self._last_used_ts = time.time()
            try:
                # Run sync transformers call off the event loop
                response = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._runtime.chat(messages, **kwargs)
                )
                self._last_used_ts = time.time()
                return response
            except Exception as e:
                self._state = ModelState.ERROR
                self._error_msg = str(e)
                logger.exception("chat() failed")
                raise

    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Sync bridge for callers running outside the FastAPI event loop."""
        if self._loop is None:
            raise RuntimeError(
                "ModelManager event loop not captured; call capture_loop() at startup."
            )
        future = asyncio.run_coroutine_threadsafe(self.chat(messages, **kwargs), self._loop)
        return future.result()

    def status(self) -> Dict:
        vram_used = 0
        if self._runtime is not None and getattr(self._runtime, "_loaded", False):
            try:
                vram_used = self._runtime._current_load_result(0).vram_used_mb
            except Exception:
                vram_used = 0
        return {
            "state": self._state.value,
            "hf_id": self.config.hf_id,
            "quant_mode": self._effective_quant_label(),
            "vram_used_mb": vram_used,
            "vram_total_mb": detect_vram_total_mb(),
            "last_used_ts": self._last_used_ts,
            "idle_seconds": self.config.idle_seconds,
            "error": self._error_msg,
        }

    # ---- idle watcher ----

    def capture_loop(self) -> None:
        """Capture the running event loop for the sync bridge. Call at app startup."""
        self._loop = asyncio.get_running_loop()

    async def start_idle_watcher(self) -> None:
        if self._watcher_task is not None:
            return
        self._watcher_task = asyncio.create_task(self._idle_watcher_loop())

    async def stop_idle_watcher(self) -> None:
        if self._watcher_task is None:
            return
        self._watcher_task.cancel()
        try:
            await self._watcher_task
        except asyncio.CancelledError:
            pass
        self._watcher_task = None

    async def _idle_watcher_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._watcher_interval)
                if self._state != ModelState.READY:
                    continue
                if time.time() - self._last_used_ts > self.config.idle_seconds:
                    logger.info("Idle threshold exceeded; unloading model")
                    async with self._lock:
                        # Re-check after acquiring lock (someone may have used it)
                        if (
                            self._state == ModelState.READY
                            and time.time() - self._last_used_ts > self.config.idle_seconds
                        ):
                            await self._unload_locked()
        except asyncio.CancelledError:
            raise

    # ---- internals (assume lock held) ----

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or exc.__class__.__name__ == "OutOfMemoryError"
        )

    async def _load_locked(self) -> None:
        if self._state == ModelState.READY and self._runtime is not None:
            return  # idempotent
        # Claim the GPU before we start allocating VRAM. This evicts any
        # other registered runtime (typically an image model). Acquire is
        # idempotent if we already hold the lock.
        GPULock.get().acquire(GPU_LOCK_NAME)
        self._error_msg = None
        self._state = ModelState.LOADING

        # Resolve quant if AUTO
        quant = self.config.quant
        if quant == QuantMode.AUTO:
            params_b = estimate_model_size_b(self.config.hf_id)
            vram_mb = detect_vram_total_mb()
            if params_b is None:
                quant = QuantMode.BF16  # safe default, fallback if not parseable
                logger.warning(f"Could not infer size from {self.config.hf_id}; defaulting to bf16")
            else:
                quant = pick_quant_for_model(params_b=params_b, vram_mb=vram_mb)
                logger.info(f"Auto-picked quant={quant.value} for {params_b}B in {vram_mb}MB VRAM")

        attempt_quant = quant
        while attempt_quant is not None:
            try:
                self._runtime = self._runtime_factory(
                    self.config.hf_id,
                    attempt_quant,
                    gguf_file=self.config.gguf_file,
                    n_ctx=self.config.n_ctx,
                )
                # Run load() in executor; capture thread id so cancel() can
                # async-raise an exception into the worker thread.
                runtime_for_call = self._runtime

                def _load_with_tid_capture():
                    self._loader_thread_id = threading.get_ident()
                    try:
                        return runtime_for_call.load()
                    finally:
                        self._loader_thread_id = None

                await asyncio.get_running_loop().run_in_executor(None, _load_with_tid_capture)
                self._state = ModelState.READY
                self._last_used_ts = time.time()
                if attempt_quant != quant:
                    logger.warning(
                        f"Loaded at fallback quant={attempt_quant.value} after OOM at {quant.value}"
                    )
                return
            except CancelledByUser:
                # User cancelled mid-load. Don't retry, don't OOM-step-down.
                # cancel() is responsible for the final state transition; here
                # just clean up the partial runtime and propagate.
                if self._runtime is not None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None, self._runtime.unload)
                    except Exception:
                        pass
                    self._runtime = None
                self._state = ModelState.UNLOADED
                self._error_msg = None
                logger.info("Load aborted: cancelled by user")
                raise
            except Exception as e:
                # Clean up any partial runtime
                if self._runtime is not None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
                    except Exception:
                        pass
                    self._runtime = None
                if self._is_oom_error(e):
                    from .gguf_utils import is_gguf_repo
                    if is_gguf_repo(self.config.hf_id):
                        logger.error(
                            f"OOM loading GGUF {self.config.hf_id}; cannot step-down "
                            f"(quant baked into file). Pick a smaller .gguf file."
                        )
                    else:
                        next_quant = self._OOM_STEP_DOWN.get(attempt_quant)
                        if next_quant is not None:
                            logger.warning(
                                f"OOM at {attempt_quant.value}; stepping down to {next_quant.value}"
                            )
                            attempt_quant = next_quant
                            continue
                # Non-OOM or no further step-down → fail
                self._state = ModelState.ERROR
                self._error_msg = str(e)
                logger.exception("load() failed")
                raise

    async def _unload_locked(self) -> None:
        if self._runtime is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
            self._runtime = None
        self._state = ModelState.UNLOADED
        self._error_msg = None
        GPULock.get().release(GPU_LOCK_NAME)

    def unload_sync(self) -> None:
        """Sync bridge so GPULock can evict the LLM from a non-async caller
        (e.g. the image runtime's worker thread).

        - If our event loop is captured (production), dispatch unload() to
          it and wait. This serialises against an in-flight chat.
        - If no loop captured (tests, pre-startup): drop the runtime
          synchronously so VRAM is freed; lock release follows naturally.
        """
        if self._loop is None:
            if self._runtime is not None:
                try:
                    self._runtime.unload()
                except Exception:
                    logger.exception("unload_sync: runtime.unload raised")
                self._runtime = None
            self._state = ModelState.UNLOADED
            self._error_msg = None
            GPULock.get().release(GPU_LOCK_NAME)
            return
        fut = asyncio.run_coroutine_threadsafe(self.unload(), self._loop)
        try:
            fut.result(timeout=30)
        except Exception:
            logger.exception("unload_sync: dispatched unload raised")

    def _effective_quant_label(self) -> str:
        if self._runtime is not None:
            return self._runtime.quant.value
        return self.config.quant.value
