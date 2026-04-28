"""Numerical correctness tests for dequant kernels.

Strategy: take a known fp16 weight → quantize via the upstream `gguf`
package → dequantize via our kernel → assert cosine similarity with
the original ≥ threshold. This catches bit-layout bugs that would
otherwise surface as "model produces gibberish" at inference time."""
import numpy as np
import pytest
import torch

from src.video_local.gguf.dequant import dequant_q4_k_s, dequant_q8_0


def _quantize_q8_0(weight_fp16: np.ndarray) -> bytes:
    """Use upstream gguf's quantizer to produce reference Q8_0 bytes."""
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q8_0).tobytes()


def _quantize_q4_k(weight_fp16: np.ndarray) -> bytes:
    """Q4_K_S uses the Q4_K storage layout (gguf's quantize gives Q4_K
    output for both Q4_K_S and Q4_K_M — the difference is in scale
    selection during quantization, not the byte layout)."""
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q4_K).tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant runs on CUDA")
class TestDequantQ8:
    def test_round_trip_cosine_similarity(self):
        torch.manual_seed(0)
        # 32-row × 64-col weight, multiple of Q8_0's block size (32)
        original = torch.randn(32, 64, dtype=torch.float16).contiguous()
        quant_bytes = _quantize_q8_0(original.numpy())

        out = dequant_q8_0(quant_bytes, shape=(32, 64))

        assert out.dtype == torch.float16
        assert out.shape == (32, 64)
        assert out.device.type == "cuda"

        # Q8_0 has very low quant error — cosine sim should be near 1
        cos = torch.nn.functional.cosine_similarity(
            original.flatten().cuda().to(torch.float32),
            out.flatten().to(torch.float32),
            dim=0,
        ).item()
        assert cos > 0.999, f"cosine_sim={cos} below 0.999 — Q8_0 dequant likely wrong"

    def test_row_major_shape(self):
        # Confirm the output shape interprets the bytes in row-major
        # order matching the GGUF spec.
        torch.manual_seed(1)
        original = torch.arange(32 * 32, dtype=torch.float16).reshape(32, 32) / 1000.0
        quant_bytes = _quantize_q8_0(original.numpy())

        out = dequant_q8_0(quant_bytes, shape=(32, 32)).cpu().to(torch.float32)
        original_f32 = original.to(torch.float32)
        # Element-wise close — Q8_0 max error is bounded
        max_err = (out - original_f32).abs().max().item()
        assert max_err < 0.05, f"max abs error {max_err} too high"


def _craft_q4k_block(d: float, dmin: float, scales: list[int], mins: list[int],
                     q4_low: list[int], q4_high: list[int]) -> bytes:
    """Hand-craft a 144-byte Q4_K super-block. Used for unit tests since
    gguf 0.18's Q4_K quantize_blocks is unimplemented (only dequant is).

    Args:
        d:       super_scale (fp16)
        dmin:    super_min (fp16)
        scales:  8 sub-block scales, each 6-bit unsigned (0..63)
        mins:    8 sub-block mins,   each 6-bit unsigned (0..63)
        q4_low:  128 nibbles (0..15) for sub-blocks 0,2,4,6 in pairs
                 — see scheme in dequant.py docstring. Effectively 4 rows
                 of 32 nibbles, each row = sub-block 2i low nibbles.
        q4_high: 128 nibbles for sub-blocks 1,3,5,7 (high nibbles same rows)
    """
    assert len(scales) == 8 and len(mins) == 8
    assert len(q4_low) == 128 and len(q4_high) == 128
    assert all(0 <= s < 64 for s in scales)
    assert all(0 <= m < 64 for m in mins)
    assert all(0 <= q < 16 for q in q4_low)
    assert all(0 <= q < 16 for q in q4_high)

    out = bytearray(144)
    out[0:2] = np.float16(d).tobytes()
    out[2:4] = np.float16(dmin).tobytes()
    # Pack scales/mins per gguf.quants.Q4_K.get_scale_min layout:
    #  bytes 0..3 (called d-bytes): scales[0..3] low 6 bits, scales[4..7] high 2 bits in upper
    #  bytes 4..7 (called m-bytes): mins[0..3]   low 6 bits, mins[4..7]   high 2 bits in upper
    #  bytes 8..11 (m_d-bytes):     low 4 = scales[4..7] low nibble, high 4 = mins[4..7] low nibble
    for i in range(4):
        sc_lo = scales[i] & 0x3F
        sc_hi_top2 = (scales[i + 4] >> 4) & 0x03  # top 2 of the 6-bit value
        out[4 + i] = sc_lo | (sc_hi_top2 << 6)

        m_lo = mins[i] & 0x3F
        m_hi_top2 = (mins[i + 4] >> 4) & 0x03
        out[8 + i] = m_lo | (m_hi_top2 << 6)

        sc_hi_lo4 = scales[i + 4] & 0x0F
        m_hi_lo4 = mins[i + 4] & 0x0F
        out[12 + i] = sc_hi_lo4 | (m_hi_lo4 << 4)

    # Pack quants: 4 rows × 32 bytes; each byte holds (low_nibble | high_nibble << 4)
    for row in range(4):
        for k in range(32):
            lo = q4_low[row * 32 + k]
            hi = q4_high[row * 32 + k]
            out[16 + row * 32 + k] = lo | (hi << 4)
    return bytes(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant runs on CUDA")
class TestDequantQ4KS:
    def test_byte_length_one_super_block(self):
        # Sanity: 144 bytes per 256 elements
        bytes_one = _craft_q4k_block(
            d=1.0, dmin=0.0,
            scales=[1] * 8, mins=[0] * 8,
            q4_low=[0] * 128, q4_high=[0] * 128,
        )
        assert len(bytes_one) == 144

    def test_dequant_zero_block_yields_zero(self):
        # All q4 = 0, dmin = 0 → all dequant values = 0 regardless of scales
        raw = _craft_q4k_block(
            d=2.0, dmin=0.0,
            scales=[5, 10, 15, 20, 25, 30, 35, 40], mins=[0] * 8,
            q4_low=[0] * 128, q4_high=[0] * 128,
        )
        out = dequant_q4_k_s(raw, shape=(1, 256))
        assert torch.allclose(out, torch.zeros(1, 256, dtype=torch.float16, device="cuda"))

    def test_dequant_matches_formula(self):
        # Construct a block where each sub-block has a unique scale and
        # uniform q4=1 within it; check x = scale_j * 1 - min_j matches.
        d = 1.0
        dmin = 1.0
        scales = [1, 2, 3, 4, 5, 6, 7, 8]
        mins = [0, 1, 2, 3, 0, 1, 2, 3]
        # For sub-block j, all 32 elements get q4=1
        # Sub-blocks 0, 2, 4, 6 are in low nibbles of rows 0..3; q4_low pattern
        # Sub-blocks 1, 3, 5, 7 are in high nibbles of rows 0..3; q4_high pattern
        q4_low = [1] * 128       # sub-blocks 0, 2, 4, 6 all q4=1
        q4_high = [1] * 128      # sub-blocks 1, 3, 5, 7 all q4=1
        raw = _craft_q4k_block(
            d=d, dmin=dmin, scales=scales, mins=mins,
            q4_low=q4_low, q4_high=q4_high,
        )
        out = dequant_q4_k_s(raw, shape=(1, 256)).cpu().to(torch.float32)
        # Sub-block j has 32 elements with value: (d * scale_j) * 1 - (dmin * min_j)
        for j, (s, m) in enumerate(zip(scales, mins)):
            expected = d * s * 1.0 - dmin * m
            actual = out[0, j * 32:(j + 1) * 32]
            assert torch.allclose(actual, torch.full_like(actual, expected), atol=1e-3), (
                f"sub-block {j}: expected {expected}, got {actual[:4].tolist()}"
            )

    def test_matches_gguf_reference_dequant(self):
        """Cross-check: our output matches gguf's Python reference dequant.
        This is the load-bearing correctness gate — if we get bit shuffling
        wrong but the formula tests pass, this catches it."""
        import gguf
        # Use the same crafted block; feed bytes to gguf reference.
        raw = _craft_q4k_block(
            d=0.5, dmin=0.25,
            scales=[7, 14, 21, 28, 35, 42, 49, 56],
            mins=[1, 2, 4, 8, 16, 32, 48, 60],
            # Vary q4 across positions
            q4_low=[(k % 16) for k in range(128)],
            q4_high=[((k * 3) % 16) for k in range(128)],
        )
        ref = gguf.quants.dequantize(
            np.frombuffer(raw, dtype=np.uint8),
            gguf.GGMLQuantizationType.Q4_K,
        ).astype(np.float32).reshape(1, 256)
        ours = dequant_q4_k_s(raw, shape=(1, 256)).cpu().to(torch.float32).numpy()
        max_err = float(np.abs(ours - ref).max())
        assert max_err < 1e-2, f"max abs error vs gguf reference: {max_err}"
