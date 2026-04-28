"""Tests for GGUFLinear — verifies it dispatches to the right dequant
kernel and applies F.linear correctly.

Q8_0: full round-trip via gguf.quants.quantize.
Q4_K_S: gguf 0.18 doesn't implement Q4_K quantize, so we verify
GGUFLinear's output equals F.linear(dequant_q4_k_s(bytes), bias) for
hand-crafted bytes — i.e. dequant correctness is already covered by
test_gguf_dequant.py's gguf-reference cross-check; this just verifies
the wrapper layer wires inputs/outputs."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.video_local.gguf.dequant import dequant_q4_k_s
from src.video_local.gguf.ops import GGUFLinear
from src.video_local.gguf.reader import GGUFTensor

# Re-use the Q4_K crafting helper from the dequant test module
from tests.video_local.test_gguf_dequant import _craft_q4k_block


def _quantize_q8_0(weight_fp16: np.ndarray) -> bytes:
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q8_0).tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguflinear_q8_matches_nn_linear():
    torch.manual_seed(0)
    in_f, out_f = 256, 64
    real_weight = torch.randn(out_f, in_f, dtype=torch.float16).contiguous()
    real_bias = torch.randn(out_f, dtype=torch.float16)

    raw = _quantize_q8_0(real_weight.numpy())
    tensor = GGUFTensor(name="w", quant_type="Q8_0", shape=(out_f, in_f),
                       raw_bytes=memoryview(raw))
    layer = GGUFLinear(weight_tensor=tensor, bias=real_bias).cuda()

    x = torch.randn(8, in_f, dtype=torch.float16).cuda()
    actual = layer(x)

    nn_layer = nn.Linear(in_f, out_f).to(torch.float16).cuda()
    with torch.no_grad():
        nn_layer.weight.copy_(real_weight)
        nn_layer.bias.copy_(real_bias)
    expected = nn_layer(x)

    assert actual.shape == expected.shape
    rel_err = (actual - expected).abs().max() / (expected.abs().max() + 1e-6)
    # Q8_0 quant error compounds through dot products; ~0.5% rel error
    # for random Gaussian inputs is typical, well below "wiring is broken".
    assert rel_err < 1e-2, f"Q8_0 GGUFLinear vs nn.Linear rel error {rel_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguflinear_q4ks_dispatches_correctly():
    """For Q4_K_S we craft bytes (gguf can't quantize Q4_K), then verify
    GGUFLinear(x) == F.linear(dequant_q4_k_s(bytes), x, bias). dequant
    correctness is covered by test_gguf_dequant.py."""
    torch.manual_seed(1)
    # 256 in_features (one super-block per output row), 4 out_features
    in_f, out_f = 256, 4
    raws = []
    for j in range(out_f):
        raws.append(_craft_q4k_block(
            d=0.5, dmin=0.1,
            scales=[2 + j, 4 + j, 6 + j, 8 + j, 10, 12, 14, 16],
            mins=[1, 2, 3, 4, 5, 6, 7, 8],
            q4_low=[(j + k) % 16 for k in range(128)],
            q4_high=[((j + k) * 2) % 16 for k in range(128)],
        ))
    raw = b"".join(raws)
    tensor = GGUFTensor(name="w", quant_type="Q4_K_S", shape=(out_f, in_f),
                       raw_bytes=memoryview(raw))
    bias = torch.randn(out_f, dtype=torch.float16)

    layer = GGUFLinear(weight_tensor=tensor, bias=bias).cuda()
    x = torch.randn(2, in_f, dtype=torch.float16).cuda()
    actual = layer(x)

    expected_weight = dequant_q4_k_s(raw, shape=(out_f, in_f))
    expected = F.linear(x, expected_weight, bias.cuda())

    assert actual.shape == expected.shape == (2, out_f)
    assert torch.allclose(actual, expected, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguflinear_no_bias():
    torch.manual_seed(0)
    in_f, out_f = 32, 16
    real_weight = torch.randn(out_f, in_f, dtype=torch.float16).contiguous()
    raw = _quantize_q8_0(real_weight.numpy())
    tensor = GGUFTensor(name="w", quant_type="Q8_0", shape=(out_f, in_f),
                       raw_bytes=memoryview(raw))

    layer = GGUFLinear(weight_tensor=tensor, bias=None).cuda()
    out = layer(torch.randn(4, in_f, dtype=torch.float16).cuda())
    assert out.shape == (4, out_f)


def test_unsupported_quant_raises():
    with pytest.raises(ValueError, match="F16"):
        # GGUFLinear doesn't support F16 (use plain nn.Linear instead)
        tensor = GGUFTensor(name="w", quant_type="F16", shape=(2, 2),
                           raw_bytes=memoryview(b"\0" * 16))
        GGUFLinear(weight_tensor=tensor, bias=None)
