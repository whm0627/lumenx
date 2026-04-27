"""GGUF repo detection + .gguf file priority for auto-pick."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional


# Quant priority for the auto-picker — best balance for typical 24GB VRAM:
# Q4_K_M is the de facto "default" quant; the rest are fallbacks if it's missing.
_QUANT_PRIORITY = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0"]


@lru_cache(maxsize=1)
def _hf_api():
    """Lazy-construct an HfApi instance (avoids import at module load)."""
    from huggingface_hub import HfApi
    return HfApi()


def is_gguf_repo(hf_id: str) -> bool:
    """Cheap heuristic: hf_id contains 'gguf' (case-insensitive)."""
    return "gguf" in hf_id.lower()


def list_gguf_files(hf_id: str) -> List[str]:
    """Network call: list .gguf files in repo via huggingface_hub.HfApi."""
    files = _hf_api().list_repo_files(hf_id)
    return [f for f in files if f.endswith(".gguf")]


def pick_default_gguf_file(files: List[str]) -> Optional[str]:
    """Pick the highest-priority quant file, or the first .gguf file as fallback."""
    for quant in _QUANT_PRIORITY:
        for f in files:
            if quant in f:
                return f
    return files[0] if files else None
