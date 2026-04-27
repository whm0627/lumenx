"""VRAM detection and automatic quantization selection."""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class QuantMode(str, Enum):
    AUTO = "auto"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "8bit"
    INT4 = "4bit"


# Approximate VRAM (MB) needed per billion parameters at each precision.
# These are inference estimates including KV cache headroom (~20%).
_VRAM_PER_B = {
    QuantMode.FP16: 2400,
    QuantMode.BF16: 2400,
    QuantMode.INT8: 1300,
    QuantMode.INT4: 700,
}


def detect_vram_total_mb() -> int:
    """Return total VRAM of device 0 in MiB, or 0 if no CUDA GPU."""
    if torch is None or not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory // (1024 * 1024))


def estimate_model_size_b(hf_id: str) -> Optional[float]:
    """Best-effort: parse the parameter count (in billions) out of a HF model id.

    Recognises forms like 'Qwen3-8B', 'Qwen3-1.7B', 'Qwen2.5-0.5B-Instruct'.
    Returns None if no '<number>B' token is found (caller should fetch HF config
    to determine size).
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:[\W_]|$)", hf_id)
    return float(match.group(1)) if match else None


def pick_quant_for_model(params_b: float, vram_mb: int) -> QuantMode:
    """Pick the highest-precision quant that fits comfortably in VRAM.

    Defaults from spec section 5.3:
      <= 4B   -> bf16
      7-8B   -> bf16
      13-14B -> 8bit
      32B    -> 4bit
      > ~50B -> reject

    If the spec default doesn't fit available VRAM, step down until it does.
    """
    if params_b <= 8:
        candidates = [QuantMode.BF16, QuantMode.INT8, QuantMode.INT4]
    elif params_b <= 16:
        candidates = [QuantMode.INT8, QuantMode.INT4]
    elif params_b <= 35:
        candidates = [QuantMode.INT4]
    else:
        raise ValueError(
            f"Model size {params_b}B is too large for the supported quant range. "
            f"Pick a smaller model or use external inference."
        )

    for q in candidates:
        needed = int(_VRAM_PER_B[q] * params_b)
        if needed <= vram_mb:
            return q

    raise ValueError(
        f"Model size {params_b}B does not fit in {vram_mb}MB VRAM at any supported quant."
    )
