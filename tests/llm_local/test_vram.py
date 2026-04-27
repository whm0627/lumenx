"""Tests for VRAM detection and auto quant strategy."""
from unittest.mock import patch

import pytest

from src.llm_local.vram import (
    QuantMode,
    detect_vram_total_mb,
    pick_quant_for_model,
    estimate_model_size_b,
)


class TestDetectVram:
    def test_returns_total_mb_when_cuda_available(self):
        with patch("src.llm_local.vram.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3
            assert detect_vram_total_mb() == 24 * 1024

    def test_returns_zero_when_no_cuda(self):
        with patch("src.llm_local.vram.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            assert detect_vram_total_mb() == 0


class TestPickQuant:
    def test_4b_picks_bf16(self):
        assert pick_quant_for_model(params_b=4.0, vram_mb=24 * 1024) == QuantMode.BF16

    def test_8b_picks_bf16(self):
        assert pick_quant_for_model(params_b=8.0, vram_mb=24 * 1024) == QuantMode.BF16

    def test_14b_picks_8bit(self):
        assert pick_quant_for_model(params_b=14.0, vram_mb=24 * 1024) == QuantMode.INT8

    def test_32b_picks_4bit(self):
        assert pick_quant_for_model(params_b=32.0, vram_mb=24 * 1024) == QuantMode.INT4

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match="too large"):
            pick_quant_for_model(params_b=70.0, vram_mb=24 * 1024)

    def test_low_vram_8b_picks_4bit(self):
        # 8B in bf16 needs ~19GB, in 8bit ~10GB. With 8GB VRAM, only 4bit (~5.6GB) fits.
        assert pick_quant_for_model(params_b=8.0, vram_mb=8 * 1024) == QuantMode.INT4

    def test_mid_vram_8b_picks_8bit(self):
        # 8B in bf16 doesn't fit in 12GB (~19GB needed) but 8bit (~10GB) does.
        assert pick_quant_for_model(params_b=8.0, vram_mb=12 * 1024) == QuantMode.INT8


class TestEstimateModelSize:
    def test_extracts_from_hf_id_with_b_suffix(self):
        assert estimate_model_size_b("Qwen/Qwen3-8B-Instruct") == 8.0
        assert estimate_model_size_b("Qwen/Qwen3-14B") == 14.0
        assert estimate_model_size_b("meta-llama/Llama-3-70B-Instruct") == 70.0

    def test_extracts_from_decimal_b(self):
        assert estimate_model_size_b("Qwen/Qwen3-1.7B") == 1.7

    def test_extracts_0_5b(self):
        assert estimate_model_size_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5

    def test_returns_none_when_no_size_in_id(self):
        assert estimate_model_size_b("gpt2") is None
