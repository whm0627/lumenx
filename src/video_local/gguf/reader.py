"""GGUF file format parser.

Wraps the upstream `gguf` package to surface only what we need:
tensor name → (raw_bytes, quant_type, shape). We don't decode tensor
data here — that's the dequant kernels' job in dequant.py.

Validates quant types against a v1 whitelist so unsupported tiers
fail early at load instead of producing garbage at inference time."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# v1 ships these three. Adding more = implementing the corresponding
# dequant kernel in dequant.py and listing it here.
SUPPORTED_QUANT_TYPES = {"F16", "Q8_0", "Q4_K_S"}


class UnsupportedQuantError(ValueError):
    """Raised when a GGUF file contains a quant_type not in SUPPORTED_QUANT_TYPES."""


@dataclass
class GGUFTensor:
    """A single tensor read from a .gguf file. raw_bytes is an mmap view —
    do NOT modify, do NOT keep a reference longer than the parent file
    handle's lifetime."""
    name: str
    quant_type: str            # "F16" | "Q8_0" | "Q4_K_S"
    shape: Tuple[int, ...]
    raw_bytes: memoryview      # mmap'd bytes, in GGUF block layout


def _gguf_type_to_name(t) -> str:
    """Map gguf.GGMLQuantizationType to our canonical name string. Pulled
    out as a function so tests can monkeypatch it to simulate unsupported
    types without authoring a malformed GGUF."""
    return t.name  # gguf.GGMLQuantizationType is an Enum; .name = "F16", etc.


def parse_gguf(path: str) -> Dict[str, GGUFTensor]:
    """Read a .gguf file and return a name → GGUFTensor dict.

    Raises UnsupportedQuantError as soon as any tensor uses a quant type
    not in SUPPORTED_QUANT_TYPES.
    """
    import gguf

    reader = gguf.GGUFReader(path)
    out: Dict[str, GGUFTensor] = {}
    for tensor in reader.tensors:
        quant_name = _gguf_type_to_name(tensor.tensor_type)
        if quant_name not in SUPPORTED_QUANT_TYPES:
            raise UnsupportedQuantError(
                f"GGUF tensor {tensor.name!r} uses quant {quant_name!r} which "
                f"is not in v1 whitelist {sorted(SUPPORTED_QUANT_TYPES)}. "
                f"To add support: implement a dequant kernel in dequant.py "
                f"and extend SUPPORTED_QUANT_TYPES."
            )
        # GGUF stores shape in column-major (innermost dim first) and as
        # numpy.uint64. Convert to row-major Python ints so callers can
        # use it directly with torch.Tensor.reshape(...).
        shape = tuple(int(d) for d in reversed(tensor.shape))
        out[tensor.name] = GGUFTensor(
            name=tensor.name,
            quant_type=quant_name,
            shape=shape,
            raw_bytes=memoryview(tensor.data),
        )
    logger.info(f"[gguf] parsed {len(out)} tensors from {path}")
    return out
