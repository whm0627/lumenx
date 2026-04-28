"""GGUFLinear — drop-in nn.Linear replacement carrying quantized
weights. Dequantizes on every forward; the dequantized fp16 tensor is
released after the F.linear call so peak VRAM stays close to the
quantized weight size."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequant_q4_k_s, dequant_q8_0
from .reader import GGUFTensor


_DEQUANT_FNS = {
    "Q8_0":   dequant_q8_0,
    "Q4_K_S": dequant_q4_k_s,
}


class GGUFLinear(nn.Module):
    """Behaves like nn.Linear(in_features, out_features) but holds its
    weight as quantized GGUF bytes. F16 is NOT supported here — use a
    plain nn.Linear with the unquantized fp16 weight instead, since
    there's no benefit to wrapping it.
    """

    def __init__(self, weight_tensor: GGUFTensor, bias: Optional[torch.Tensor]):
        super().__init__()
        if weight_tensor.quant_type not in _DEQUANT_FNS:
            raise ValueError(
                f"GGUFLinear got unsupported quant_type {weight_tensor.quant_type!r}; "
                f"supported: {sorted(_DEQUANT_FNS)}"
            )
        self._dequant = _DEQUANT_FNS[weight_tensor.quant_type]
        self._shape = weight_tensor.shape
        # Store raw bytes as a buffer so .to(device) moves them along
        # with the rest of the module. uint8 is the natural type for
        # opaque packed bytes.
        raw = torch.frombuffer(bytearray(weight_tensor.raw_bytes), dtype=torch.uint8)
        self.register_buffer("_quant_bytes", raw, persistent=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float16))
        else:
            self.register_parameter("bias", None)

        # Convenience attrs for callers that introspect Linear shape
        self.out_features, self.in_features = self._shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass raw bytes to dequant; it handles device placement.
        # NOTE: the dequant fns currently take a Python bytes/memoryview
        # and do their own H2D copy. Keeping that interface for now
        # so dequant tests stay simple. A follow-up could push the
        # bytes onto the GPU once and let dequant work on a device tensor.
        raw_bytes = bytes(self._quant_bytes.cpu().numpy())
        weight = self._dequant(raw_bytes, shape=self._shape, out_device=x.device.type)
        return F.linear(x, weight, self.bias)
