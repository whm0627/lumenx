"""Read live VRAM usage via nvidia-smi for the global status footer.

This is intentionally never-raises: returns zeros on any error so a UI
poll doesn't fail when the driver hiccups, the binary's missing, or the
machine has no NVIDIA GPU at all.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Dict

logger = logging.getLogger(__name__)

_NVIDIA_SMI_ARGS = [
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def read_gpu_stats() -> Dict[str, int]:
    """Return {used_mb, total_mb}. Zero values indicate "no data available"."""
    try:
        proc = subprocess.run(
            _NVIDIA_SMI_ARGS,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except FileNotFoundError:
        return {"used_mb": 0, "total_mb": 0}
    except Exception:
        logger.exception("read_gpu_stats: subprocess.run raised")
        return {"used_mb": 0, "total_mb": 0}

    if proc.returncode != 0:
        return {"used_mb": 0, "total_mb": 0}

    try:
        first_line = proc.stdout.strip().splitlines()[0]
        used_str, total_str = (s.strip() for s in first_line.split(","))
        return {"used_mb": int(used_str), "total_mb": int(total_str)}
    except Exception:
        return {"used_mb": 0, "total_mb": 0}
