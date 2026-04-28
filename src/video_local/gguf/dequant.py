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


_Q4_K_SUPER_BLOCK_SIZE = 256
_Q4_K_BYTES = 144           # per super-block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
_Q4_K_SCALE_BYTES = 12      # K_SCALE_SIZE in gguf reference


def dequant_q4_k_s(
    raw: bytes | memoryview,
    shape: tuple[int, ...],
    out_device: str = "cuda",
) -> torch.Tensor:
    """Dequantize Q4_K_S bytes to fp16 tensor of `shape` on `out_device`.

    Q4_K_S uses the Q4_K block layout (Q4_K_S vs Q4_K_M differ only in
    quantization-time scale-rounding heuristics, not in storage).

    Per super-block (256 elements, 144 bytes):
      [0..2]    fp16 super_scale d
      [2..4]    fp16 super_min   dmin
      [4..16]   12 bytes packed scale/min — see _q4k_unpack_scale_min
      [16..144] 128 bytes packed 4-bit quants for 8 sub-blocks of 32 elements

    For each of the 8 sub-blocks (j = 0..7):
      scale_j = d * sc_j        # where sc_j is 6-bit unsigned
      min_j   = dmin * m_j
      x[j*32 + k] = scale_j * q4[j*32 + k] - min_j

    The 12-byte scale/min encoding (canonical gguf layout, see
    gguf.quants.Q4_K.get_scale_min for reference):
      bytes 0..3 : 6-bit scales[0..3] in low 6 bits, high 2 bits go to scales[4..7]
      bytes 4..7 : 6-bit mins[0..3]   in low 6 bits, high 2 bits go to mins[4..7]
      bytes 8..11: low 4 bits = scale[4..7] low nibble; high 4 bits = min[4..7] low nibble
      sc[4..7] = (m_d_lo & 0x0F) | ((d >> 2) & 0x30)
      m[4..7]  = (m_d_hi)         | ((m >> 2) & 0x30)
    """
    n = 1
    for d in shape:
        n *= d
    assert n % _Q4_K_SUPER_BLOCK_SIZE == 0, (
        f"Q4_K_S requires total elements multiple of {_Q4_K_SUPER_BLOCK_SIZE}, got {n}"
    )
    n_super = n // _Q4_K_SUPER_BLOCK_SIZE
    expected_bytes = n_super * _Q4_K_BYTES
    assert len(raw) == expected_bytes, (
        f"Q4_K_S byte length mismatch: expected {expected_bytes}, got {len(raw)}"
    )

    blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(out_device)
    blob = blob.view(n_super, _Q4_K_BYTES)

    # Super-scale and super-min
    super_d = blob[:, 0:2].contiguous().view(torch.float16).squeeze(-1)    # (n_super,)
    super_m = blob[:, 2:4].contiguous().view(torch.float16).squeeze(-1)    # (n_super,)

    # 12-byte packed scales/mins → 8 6-bit scales + 8 6-bit mins
    sc, mn = _q4k_unpack_scale_min(blob[:, 4:16])                          # each (n_super, 8) uint8

    # Per-sub-block scale and min (fp32 to avoid overflow in scale * min)
    super_d_f32 = super_d.to(torch.float32).unsqueeze(-1)                  # (n_super, 1)
    super_m_f32 = super_m.to(torch.float32).unsqueeze(-1)                  # (n_super, 1)
    scale = (super_d_f32 * sc.to(torch.float32))                           # (n_super, 8)
    minv  = (super_m_f32 * mn.to(torch.float32))                           # (n_super, 8)

    # 128-byte quants — reshape as 4 rows of 32 bytes; each row low/high
    # nibbles produce 2 sub-blocks of 32 elements each (8 sub-blocks total).
    q_pack = blob[:, 16:].view(n_super, 4, 32).to(torch.int32)             # (n_super, 4, 32)
    # Stack low and high nibble axis: (n_super, 4, 2, 32)
    q_lo = q_pack & 0x0F                                                   # low nibble = sub-block 2i
    q_hi = (q_pack >> 4) & 0x0F                                            # high nibble = sub-block 2i+1
    q_pairs = torch.stack([q_lo, q_hi], dim=2)                             # (n_super, 4, 2, 32)
    q4 = q_pairs.reshape(n_super, 8, 32).to(torch.float32)                 # (n_super, 8, 32)

    out = q4 * scale.unsqueeze(-1) - minv.unsqueeze(-1)                    # (n_super, 8, 32)
    return out.reshape(shape).to(torch.float16).contiguous()


def _q4k_unpack_scale_min(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the 12-byte Q4_K scale/min block into two (n_super, 8) uint8
    tensors of 6-bit values, matching gguf.quants.Q4_K.get_scale_min.

    The 12 bytes split into 3 groups of 4:
      d   = pack[:, 0:4]   — 4 bytes encoding scales[0..3] (low 6) + scales[4..7] high 2 bits
      m   = pack[:, 4:8]   — 4 bytes encoding mins[0..3]   (low 6) + mins[4..7]   high 2 bits
      m_d = pack[:, 8:12]  — 4 bytes: low 4 bits = scales[4..7] low nibble;
                                       high 4 bits = mins[4..7]   low nibble
    """
    d = pack[:, 0:4]
    m = pack[:, 4:8]
    m_d = pack[:, 8:12]
    sc_low = d & 0x3F                                       # scales[0..3]
    sc_high = (m_d & 0x0F) | ((d >> 2) & 0x30)              # scales[4..7]
    mn_low = m & 0x3F                                       # mins[0..3]
    mn_high = (m_d >> 4) | ((m >> 2) & 0x30)                # mins[4..7]
    sc = torch.cat([sc_low, sc_high], dim=1)                # (n_super, 8)
    mn = torch.cat([mn_low, mn_high], dim=1)                # (n_super, 8)
    return sc, mn
