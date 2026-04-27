"""EmbeddedServerRuntime — runs llama-server.exe as a subprocess.

Drop-in replacement for GGUFRuntime, with the same interface (load/chat/unload
+ LoadResult). Differences:

- Out-of-process: VRAM is freed cleanly when the subprocess dies (no CUDA
  context leak, no lingering Python references).
- Always uses upstream llama.cpp's binary (auto-updated by LlamaCppManager).
- Communicates via OpenAI-compatible HTTP on localhost.

Concurrency: ModelManager already serialises load/chat/unload via its lock,
so this class is intentionally single-threaded internally.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from openai import OpenAI

from .runtime import LoadResult
from .server_proc import LlamaServerProcess
from .vram import QuantMode

logger = logging.getLogger(__name__)

# Force stderr to UTF-8 so Chinese tokens streamed during chat() display
# correctly in the dev cmd window (default Windows codepage is cp936/GBK
# which mangles UTF-8 bytes into mojibake). Pair this with `chcp 65001`
# in the spawning cmd or text will still look garbled even if bytes are right.
import sys as _sys
try:
    _sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, Exception):  # noqa: BLE001 — best-effort
    pass


# Singleton accessor — set by api.py at startup so runtime_server doesn't
# need to know the runtime_parent path.
_BINARY_MANAGER = None


def get_binary_manager():
    """Return the LlamaCppManager singleton. Set via set_binary_manager()."""
    if _BINARY_MANAGER is None:
        # Lazy fallback: create a default manager rooted at output/runtime/
        from pathlib import Path as _P
        from .llama_cpp_release import LlamaCppManager
        proj_root = _P(__file__).resolve().parents[2]
        return LlamaCppManager(runtime_parent=proj_root / "output" / "runtime")
    return _BINARY_MANAGER


def set_binary_manager(mgr):
    """Install the singleton (called by api.py startup)."""
    global _BINARY_MANAGER
    _BINARY_MANAGER = mgr


# Default port for the embedded server. Adjacent to lumenx backend's 17177.
DEFAULT_SERVER_PORT = 17178


class EmbeddedServerRuntime:
    """Runs llama-server.exe as a subprocess and chats via its OpenAI API."""

    def __init__(self, hf_id: str, gguf_file: str, port: int = DEFAULT_SERVER_PORT,
                 n_ctx: int = 8192, startup_timeout: float = 120.0):
        self.hf_id = hf_id
        self.gguf_file = gguf_file
        self.port = port
        self.n_ctx = n_ctx
        self.startup_timeout = startup_timeout
        self.quant = QuantMode.AUTO  # not meaningful for GGUF; interface parity
        self._proc: Optional[LlamaServerProcess] = None
        self._loaded = False

    def load(self) -> LoadResult:
        """Idempotent. Verify binary, fetch GGUF, spawn server, wait for healthy."""
        if self._loaded and self._proc is not None and self._proc.is_alive():
            return self._current_load_result(elapsed_sec=0.0)

        t0 = time.monotonic()

        # 1. Resolve llama-server.exe path
        mgr = get_binary_manager()
        exe = mgr.current_exe_path()
        if exe is None:
            raise RuntimeError(
                "llama.cpp runtime not installed. The backend should auto-download "
                "it on startup; if you just started the backend, give it ~1 minute "
                "and try again. If GitHub is unreachable, manually drop "
                "llama-server.exe + DLLs into output/runtime/llama.cpp-<tag>/."
            )

        # 2. Resolve gguf path (download if not cached; uses HF_HOME)
        logger.info(f"Resolving GGUF: {self.hf_id} :: {self.gguf_file}")
        gguf_path = Path(hf_hub_download(repo_id=self.hf_id, filename=self.gguf_file))

        # 3. Spawn server
        logger.info(f"Starting llama-server pointing at {gguf_path.name}")
        self._proc = LlamaServerProcess(
            exe=exe,
            gguf_path=gguf_path,
            port=self.port,
            n_ctx=self.n_ctx,
        )
        self._proc.start(startup_timeout=self.startup_timeout)
        self._loaded = True

        elapsed = time.monotonic() - t0
        logger.info(f"EmbeddedServerRuntime ready in {elapsed:.1f}s")
        return self._current_load_result(elapsed_sec=elapsed)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 16384,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """Chat via the embedded llama-server's OpenAI-compatible endpoint.

        max_new_tokens default is 16384 (not 1024) because Qwen3 family models
        burn 100s–1000s of tokens on internal <think> reasoning before any
        visible output, and script-extraction JSON for a full comic can itself
        run several thousand tokens. 1024 was empirically not enough.

        When response_format requests JSON, we also pass
        chat_template_kwargs={"enable_thinking": false} via OpenAI extra_body
        so Qwen3 skips the thinking phase entirely — observed ~86x reduction in
        completion_tokens (2 vs 172 for "what is 2+2").
        """
        if not self._loaded or self._proc is None:
            raise RuntimeError("EmbeddedServerRuntime not loaded; call load() first.")

        client = OpenAI(
            api_key="not-needed",  # llama-server doesn't validate
            base_url=f"{self._proc.base_url}/v1",
        )

        kwargs: Dict[str, object] = {
            "model": "local",  # llama-server ignores this; uses the loaded model
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        extra_body: Dict[str, object] = {}

        if response_format is not None:
            kwargs["response_format"] = response_format
            # JSON output requested → disable Qwen3 thinking. The reasoning
            # phase doesn't help here (the model is constrained to emit JSON
            # anyway) and burning tokens on it can starve the JSON output.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Stream so the backend cmd window shows thinking tokens live (Qwen3
        # emits <think>...</think> as regular content tokens). Concatenate
        # chunks and return the same string a non-stream call would give.
        kwargs["stream"] = True
        logger.info(f"[stream] starting (json_mode={response_format is not None})")
        _sys.stderr.write("\n--- stream begin ---\n"); _sys.stderr.flush()
        parts: List[str] = []
        n_chunks = 0
        for chunk in client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                # stderr is line-unbuffered on Windows; print() to stdout in a
                # uvicorn worker thread under concurrently can buffer indefinitely.
                _sys.stderr.write(delta); _sys.stderr.flush()
                parts.append(delta)
                n_chunks += 1
        _sys.stderr.write(f"\n--- stream end (chunks={n_chunks}, chars={sum(len(p) for p in parts)}) ---\n")
        _sys.stderr.flush()
        logger.info(f"[stream] complete chunks={n_chunks} chars={sum(len(p) for p in parts)}")
        return "".join(parts)

    def unload(self) -> None:
        """Idempotent. Terminates the subprocess (frees VRAM via OS)."""
        if self._proc is not None:
            logger.info(f"Terminating llama-server for {self.hf_id}")
            self._proc.terminate()
            self._proc = None
        self._loaded = False

    def _current_load_result(self, elapsed_sec: float) -> LoadResult:
        # llama-server doesn't expose VRAM usage to us via its API; report 0.
        return LoadResult(
            hf_id=self.hf_id,
            quant_mode=self.quant,
            vram_used_mb=0,
            elapsed_sec=elapsed_sec,
        )
