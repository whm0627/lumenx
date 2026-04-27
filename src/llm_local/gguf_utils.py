"""GGUF repo detection + .gguf file priority for auto-pick."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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
    """Return .gguf filenames in the repo.

    Local-first: if the model is already cached on disk, use that — no
    network roundtrip, works offline, and survives the case where the
    upstream repo flipped to private after a successful download.
    Only consults the HF API when nothing's cached locally."""
    local = _list_local_gguf_files(hf_id)
    if local:
        return local
    files = _hf_api().list_repo_files(hf_id)
    return [f for f in files if f.endswith(".gguf")]


def _list_local_gguf_files(hf_id: str) -> List[str]:
    """Scan the HF cache for .gguf files belonging to `hf_id`. Returns
    filenames (not full paths). Empty list if no cache directory exists."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return []
    repo_dir = Path(hf_home) / "hub" / ("models--" + hf_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return []
    # Pick the latest snapshot revision and list .gguf files in it.
    snaps = [s for s in snapshots.iterdir() if s.is_dir()]
    if not snaps:
        return []
    latest = max(snaps, key=lambda p: p.stat().st_mtime)
    return sorted(f.name for f in latest.iterdir() if f.is_file() and f.name.endswith(".gguf"))


def pick_default_gguf_file(files: List[str]) -> Optional[str]:
    """Pick the highest-priority quant file, or the first .gguf file as fallback."""
    for quant in _QUANT_PRIORITY:
        for f in files:
            if quant in f:
                return f
    return files[0] if files else None
