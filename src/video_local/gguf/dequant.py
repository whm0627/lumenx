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


def _to_uint8_blob(raw, out_device: str) -> torch.Tensor:
    """Normalize input to a uint8 tensor on `out_device`.

    Accepts (bytes/memoryview, on host) or (torch.Tensor uint8, on any
    device). The tensor path skips the `frombuffer + to(device)` round-
    trip — used by GGUFLinear which keeps quant bytes resident on GPU
    as a buffer.
    """
    if isinstance(raw, torch.Tensor):
        if raw.dtype != torch.uint8:
            raw = raw.to(torch.uint8)
        if raw.device.type != out_device:
            raw = raw.to(out_device)
        return raw.contiguous()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(out_device)


def dequant_q8_0(
    raw,  # bytes | memoryview | torch.Tensor (uint8)
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

    blob = _to_uint8_blob(raw, out_device)
    expected_bytes = n_blocks * _Q8_0_BYTES
    assert blob.numel() == expected_bytes, (
        f"Q8_0 byte length mismatch: expected {expected_bytes}, got {blob.numel()}"
    )
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
    raw,  # bytes | memoryview | torch.Tensor (uint8)
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

    blob = _to_uint8_blob(raw, out_device)
    assert blob.numel() == expected_bytes, (
        f"Q4_K_S byte length mismatch: expected {expected_bytes}, got {blob.numel()}"
    )
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


_Q5_K_BYTES = 176     # 2(d) + 2(dmin) + 12(scales) + 32(qh) + 128(qs)


def dequant_q5_k(
    raw,
    shape: tuple[int, ...],
    out_device: str = "cuda",
) -> torch.Tensor:
    """Dequantize Q5_K bytes to fp16 tensor of `shape` on `out_device`.

    Q5_K extends Q4_K with one extra high bit per element. Each
    super-block stores 256 elements as 8 sub-blocks of 32. The scale
    and min encoding is identical to Q4_K (12 bytes per super-block),
    but quants are 5 bits each: low 4 bits in 128 bytes of `qs`, high
    1 bit in 32 bytes of `qh` (bit-packed across 8 sub-blocks).

    Layout (176 bytes per super-block):
      [0..2]    fp16 super_scale d
      [2..4]    fp16 super_min   dmin
      [4..16]   12 bytes packed 6-bit scales/mins  (same as Q4_K)
      [16..48]  32 bytes qh — bit `j` of byte `i` is high bit of
                element `i` in sub-block `j` (0 ≤ j < 8)
      [48..176] 128 bytes qs — packed 4-bit low nibbles, two
                sub-blocks per row (low/high nibble)

    Mixed-strategy GGUFs (e.g. QuantStack's Q4_K_S file) use Q5_K for
    attention V projections + FFN.2 weights, so we MUST handle this
    quant type even when the file name says Q4_K_S.
    """
    n = 1
    for d in shape:
        n *= d
    assert n % _Q4_K_SUPER_BLOCK_SIZE == 0, (
        f"Q5_K requires total elements multiple of {_Q4_K_SUPER_BLOCK_SIZE}, got {n}"
    )
    n_super = n // _Q4_K_SUPER_BLOCK_SIZE
    expected_bytes = n_super * _Q5_K_BYTES

    blob = _to_uint8_blob(raw, out_device)
    assert blob.numel() == expected_bytes, (
        f"Q5_K byte length mismatch: expected {expected_bytes}, got {blob.numel()}"
    )
    blob = blob.view(n_super, _Q5_K_BYTES)

    super_d = blob[:, 0:2].contiguous().view(torch.float16).squeeze(-1)
    super_m = blob[:, 2:4].contiguous().view(torch.float16).squeeze(-1)
    sc, mn = _q4k_unpack_scale_min(blob[:, 4:16])

    super_d_f32 = super_d.to(torch.float32).unsqueeze(-1)
    super_m_f32 = super_m.to(torch.float32).unsqueeze(-1)
    scale = super_d_f32 * sc.to(torch.float32)              # (n_super, 8)
    minv = super_m_f32 * mn.to(torch.float32)               # (n_super, 8)

    # Low nibbles: 128 bytes → 8 sub-blocks of 32 elements (low/high
    # nibble per byte), same unpacking as Q4_K.
    qs = blob[:, 48:176].contiguous()
    q_pack = qs.view(n_super, 4, 32).to(torch.int32)        # (n_super, 4, 32)
    q_lo = q_pack & 0x0F
    q_hi_n = (q_pack >> 4) & 0x0F
    q_pairs = torch.stack([q_lo, q_hi_n], dim=2)            # (n_super, 4, 2, 32)
    q4 = q_pairs.reshape(n_super, 8, 32)                    # (n_super, 8, 32)

    # High bits: 32 bytes → 8 sub-blocks × 32 elements, 1 bit each.
    # Bit `j` of byte `i` = high bit of element `i` in sub-block `j`.
    qh = blob[:, 16:48].contiguous()
    qh_bits = qh.view(n_super, 1, 1, 32).to(torch.int32) >> \
        torch.arange(0, 8, dtype=torch.int32, device=qh.device).view(1, 1, 8, 1)
    qh_bits = (qh_bits & 0x01).view(n_super, 8, 32)         # (n_super, 8, 32)

    q5 = (q4 | (qh_bits << 4)).to(torch.float32)            # 0..31

    out = q5 * scale.unsqueeze(-1) - minv.unsqueeze(-1)     # (n_super, 8, 32)
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
