"""VideoModelManager — singleton state machine wrapping LocalWanS2V.

Tracks the active quant tier so the footer can show
"VIDEO Wan2.2-S2V Q4_K_S" and switching tiers triggers an unload + load
(no shared weight state).

Two `generate()` shapes are exposed:

  - `generate_native(image, audio, prompt, output_path, **kw)` — direct
    passthrough for /video/local/test_synthesize. Local-only.

  - `generate(prompt, output_path, img_path, **kwargs)` — cloud
    WanxModel.generate-compatible signature. Lets pipeline.py call us
    the same way it calls the cloud model. Maps `audio_url` /
    `audio_path` strings to the Wan-S2V audio driver; ignores the cloud
    `audio: bool` kwarg (which on the cloud means "synthesize TTS for
    me" — locally that's the SFX/TTS layer's job, upstream of us).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .wan_s2v_runtime import LocalWanS2V, _SUPPORTED_QUANTS
from ..utils.gpu_lock import GPULock

logger = logging.getLogger(__name__)
GPU_LOCK_NAME = "video"


class VideoState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"
    LOADING = "LOADING"
    GENERATING = "GENERATING"
    READY = "READY"
    ERROR = "ERROR"


_DEFAULT_QUANT = os.environ.get("LOCAL_VIDEO_QUANT", "Q4_K_S")


class VideoModelManager:
    _instance: Optional["VideoModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[Dict[str, Any]] = None) -> "VideoModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VideoModelManager(config or {})
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._quant = _DEFAULT_QUANT
        self._inner = LocalWanS2V(quant=self._quant)
        self._state = VideoState.UNLOADED
        self._error: Optional[str] = None
        self._loaded = False
        self._gen_progress = 0.0
        self._phase_label = ""
        GPULock.get().register(GPU_LOCK_NAME, self.unload_sync)

    def status(self) -> Dict[str, Any]:
        progress = (
            self._gen_progress if self._state == VideoState.GENERATING
            else (1.0 if self._state == VideoState.READY else 0.0)
        )
        return {
            "state": self._state.value,
            "quant": self._quant,
            "hf_id": self._inner.hf_id,
            "phase": "",
            "progress": progress,
            "error": self._error,
            "phase_label": self._phase_label,
        }

    async def load(self, quant: Optional[str] = None) -> Dict[str, Any]:
        target = quant or self._quant
        if target not in _SUPPORTED_QUANTS:
            raise ValueError(
                f"Unsupported local video quant {target!r}; expected "
                f"{sorted(_SUPPORTED_QUANTS)}"
            )
        if self._state == VideoState.READY and target == self._quant:
            return self.status()
        if self._loaded and target != self._quant:
            await asyncio.to_thread(self._inner.unload)
            self._loaded = False
        self._quant = target
        self._inner.quant = target
        self._state = VideoState.LOADING
        self._error = None
        try:
            GPULock.get().acquire(GPU_LOCK_NAME)
            await asyncio.to_thread(self._inner.load)
            self._loaded = True
            self._state = VideoState.READY
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.load failed")
            raise
        return self.status()

    async def unload(self) -> None:
        await asyncio.to_thread(self._inner.unload)
        self._state = VideoState.UNLOADED
        self._error = None
        self._loaded = False
        GPULock.get().release(GPU_LOCK_NAME)

    def unload_sync(self) -> None:
        """Synchronous GPULock eviction hook."""
        try:
            self._inner.unload()
        finally:
            self._state = VideoState.UNLOADED
            self._error = None
            self._loaded = False
            self._gen_progress = 0.0
            self._phase_label = ""
            GPULock.get().release(GPU_LOCK_NAME)

    # ---- generation -------------------------------------------------

    def generate_native(self, image: str, audio: Optional[str], prompt: str,
                        output_path: str, **kwargs) -> str:
        """Direct passthrough — used by /video/local/test_synthesize."""
        if not self._loaded:
            self._state = VideoState.LOADING
            try:
                GPULock.get().acquire(GPU_LOCK_NAME)
                self._inner.load()
                self._loaded = True
            except Exception as e:
                self._state = VideoState.ERROR
                self._error = str(e)
                raise
        GPULock.get().acquire(GPU_LOCK_NAME)
        self._state = VideoState.GENERATING
        self._gen_progress = 0.0
        try:
            result = self._inner.generate(
                image=image, audio=audio, prompt=prompt,
                output_path=output_path, **kwargs,
            )
            self._state = VideoState.READY
            self._gen_progress = 1.0
            self._error = None
            return result
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.generate_native failed")
            raise

    @staticmethod
    def _audio_path(*candidates) -> Optional[str]:
        """Pick the first candidate that's a non-empty string. NOTE the
        cloud WanxModel.generate signature also has an `audio` kwarg,
        but on the cloud side it's a *boolean* meaning "let the backend
        generate TTS for me" — pipeline.py wires `audio=final_generate_audio`
        (a bool). We only accept path-shaped strings here; booleans and
        Nones fall through to the silent-WAV fallback in the runtime."""
        for c in candidates:
            if isinstance(c, str) and c:
                return c
        return None

    def generate(self, prompt: str, output_path: str, img_path: Optional[str] = None,
                 **kwargs) -> Tuple[str, float]:
        """Cloud-WanxModel-compatible signature. Returns (path, duration_sec).

        Maps cloud kwargs to LocalWanS2V.generate:
          img_path → image (required)
          img_url  → fetched to a temp local path if img_path is None
          audio_url / audio_path / audio → driver wav path (string only)
          duration → frame_num via fps=16, snapped to 4n
          negative_prompt / seed / max_area / sampling_steps / guide_scale /
            shift → forwarded
        """
        if not self._loaded:
            self._state = VideoState.LOADING
            try:
                GPULock.get().acquire(GPU_LOCK_NAME)
                self._inner.load()
                self._loaded = True
            except Exception as e:
                self._state = VideoState.ERROR
                self._error = str(e)
                raise
        GPULock.get().acquire(GPU_LOCK_NAME)

        image = img_path or kwargs.get("image")
        if not image:
            img_url = kwargs.get("img_url")
            if not img_url:
                raise ValueError(
                    "Local Wan2.2-S2V requires an input image: pass img_path, "
                    "image, or img_url"
                )
            image = self._download_to_temp(img_url)

        audio = self._audio_path(
            kwargs.get("audio_url"), kwargs.get("audio_path"), kwargs.get("audio"),
        )

        seed = kwargs.get("seed")
        forward_kwargs: Dict[str, Any] = {}
        if isinstance(seed, int) and seed >= 0:
            forward_kwargs["seed"] = seed
        if kwargs.get("negative_prompt"):
            forward_kwargs["negative_prompt"] = kwargs["negative_prompt"]
        for key in ("max_area", "sampling_steps", "guide_scale", "shift"):
            if kwargs.get(key) is not None:
                forward_kwargs[key] = kwargs[key]

        # Map duration (seconds) → frame_num at S2V's sample_fps=16,
        # snapped to 4n (Wan's S2V hard requirement).
        fps = 16
        duration_sec = float(kwargs.get("duration") or 5)
        frame_num = max(4, (int(round(duration_sec * fps)) // 4) * 4)
        forward_kwargs["frame_num"] = frame_num

        self._state = VideoState.GENERATING
        self._gen_progress = 0.0
        try:
            path = self._inner.generate(
                image=image, audio=audio, prompt=prompt,
                output_path=output_path, **forward_kwargs,
            )
            self._state = VideoState.READY
            self._gen_progress = 1.0
            self._error = None
            return path, frame_num / fps
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.generate failed")
            raise

    @staticmethod
    def _download_to_temp(url: str) -> str:
        import tempfile
        import urllib.parse
        import urllib.request

        suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".png"
        fd, tmp = tempfile.mkstemp(prefix="wan_s2v_input_", suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(url, tmp)
        return tmp
