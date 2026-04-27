"""GGUFRuntime — synchronous llama-cpp-python wrapper for one GGUF model at a time."""
from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Optional

# Import torch here to populate Windows DLL search path with bundled CUDA libs
# (cudart64_12.dll, cublas64_12.dll). llama-cpp-python's CUDA wheels reference
# these but don't bundle them; without torch loaded first, llama.dll fails to
# load on systems whose CUDA toolkit version differs from the wheel's target.
import torch  # noqa: F401

from .runtime import LoadResult
from .vram import QuantMode

logger = logging.getLogger(__name__)


class GGUFRuntime:
    """Owns a single loaded GGUF model. Pure sync; ModelManager owns concurrency.

    Implements the same minimal interface as LocalRuntime:
      - load() -> LoadResult
      - chat(messages, max_new_tokens=, temperature=) -> str
      - unload() -> None
    """

    def __init__(self, hf_id: str, gguf_file: str, n_ctx: int = 8192):
        self.hf_id = hf_id
        self.gguf_file = gguf_file
        self.n_ctx = n_ctx
        self.quant = QuantMode.AUTO  # not meaningful for GGUF; kept for interface parity
        self._llama = None
        self._loaded = False

    def load(self) -> LoadResult:
        """Idempotent. Downloads via hf_hub_download (uses HF_HOME cache) and loads via llama_cpp."""
        if self._loaded:
            return self._current_load_result(elapsed_sec=0.0)

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        t0 = time.monotonic()
        logger.info(f"Loading GGUF {self.hf_id}/{self.gguf_file} (n_ctx={self.n_ctx})")

        # hf_hub_download honours HF_HOME (set by src/llm_local/__init__.py)
        local_path = hf_hub_download(repo_id=self.hf_id, filename=self.gguf_file)

        self._llama = Llama(
            model_path=local_path,
            n_gpu_layers=-1,        # offload all layers to GPU
            n_ctx=self.n_ctx,
            verbose=False,
            chat_format=None,       # auto-detect from GGUF metadata
        )
        self._loaded = True

        elapsed = time.monotonic() - t0
        logger.info(f"Loaded GGUF {self.hf_id} in {elapsed:.1f}s")
        return self._current_load_result(elapsed_sec=elapsed)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("GGUF runtime not loaded; call load() first.")

        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"]

    def unload(self) -> None:
        if not self._loaded and self._llama is None:
            return
        logger.info(f"Unloading GGUF {self.hf_id}")
        # llama_cpp.Llama frees its internal context on __del__
        del self._llama
        self._llama = None
        self._loaded = False
        gc.collect()

    def _current_load_result(self, elapsed_sec: float) -> LoadResult:
        # llama_cpp does not expose VRAM usage directly; report 0 (documented limitation).
        return LoadResult(
            hf_id=self.hf_id,
            quant_mode=self.quant,
            vram_used_mb=0,
            elapsed_sec=elapsed_sec,
        )
