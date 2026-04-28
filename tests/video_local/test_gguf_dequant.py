"""Numerical correctness tests for dequant kernels.

Strategy: take a known fp16 weight → quantize via the upstream `gguf`
package → dequantize via our kernel → assert cosine similarity with
the original ≥ threshold. This catches bit-layout bugs that would
otherwise surface as "model produces gibberish" at inference time."""
import numpy as np
import pytest
import torch

from src.video_local.gguf.dequant import dequant_q8_0


def _quantize_q8_0(weight_fp16: np.ndarray) -> bytes:
    """Use upstream gguf's quantizer to produce reference Q8_0 bytes."""
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q8_0).tobytes()


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
