"""GGUF dequantization kernels — torch ops, fp16 output, CUDA-resident.

Each block layout follows the GGUF/GGML spec. References:
- llama.cpp ggml-quants.c (canonical implementation)
- city96/ComfyUI-GGUF/dequant.py (PyTorch port we're paralleling)

Q8_0 layout (per 32-element block):
  bytes [0..2]   : fp16 scale `d`
  bytes [2..34]  : 32 int8 quantized values `qs`
  total: 34 bytes per block, dequantized as: x = qs * d

Q4_K_S layout (per 256-element super-block):
  bytes [0..2]   : fp16 super_scale `d`
  bytes [2..4]   : fp16 super_min `dmin`
  bytes [4..16]  : 12 bytes packed 6-bit scales/mins for 16 sub-blocks
                   of 16 elements each
  bytes [16..144]: 128 bytes packed 4-bit quantized values
  total: 144 bytes per super-block. Dequant per sub-block i (16 elements):
    scale_i  = d * (scales_packed[i] & 0x3F)
    min_i    = dmin * (mins_packed[i] & 0x3F)
    x[i*16 + j] = scale_i * q4[i*16 + j] - min_i
"""
from __future__ import annotations

import torch


_Q8_0_BLOCK_SIZE = 32          # elements per Q8_0 block
_Q8_0_BYTES = 34               # bytes per Q8_0 block (2 fp16 scale + 32 int8)


def dequant_q8_0(
    raw: bytes | memoryview,
    shape: tuple[int, ...],
    out_device: str = "cuda",
) -> torch.Tensor:
    """Dequantize Q8_0 bytes to fp16 tensor of `shape` on `out_device`.

    Q8_0 stores N elements as N/32 blocks; each block is 1 fp16 scale +
    32 int8 quantized values. dequant: x = qs * scale.
    """
    n = 1
    for d in shape:
        n *= d
    assert n % _Q8_0_BLOCK_SIZE == 0, (
        f"Q8_0 requires total elements multiple of {_Q8_0_BLOCK_SIZE}, got {n}"
    )
    n_blocks = n // _Q8_0_BLOCK_SIZE
    expected_bytes = n_blocks * _Q8_0_BYTES
    assert len(raw) == expected_bytes, (
        f"Q8_0 byte length mismatch: expected {expected_bytes}, got {len(raw)}"
    )

    # Move bytes to GPU as a contiguous uint8 tensor — single H2D copy
    blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(out_device)
    blob = blob.view(n_blocks, _Q8_0_BYTES)

    # First 2 bytes of each block are fp16 scale; remaining 32 are int8 qs.
    scales = blob[:, :2].contiguous().view(torch.float16)         # (n_blocks, 1)
    qs = blob[:, 2:].view(torch.int8).to(torch.float16)            # (n_blocks, 32)
    dequant = qs * scales                                          # (n_blocks, 32)

    return dequant.reshape(shape).contiguous()
