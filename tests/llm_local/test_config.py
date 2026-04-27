"""Tests for LocalLLMConfig persistence."""
from unittest.mock import patch

from src.llm_local.config import LocalLLMConfig
from src.llm_local.vram import QuantMode


class TestLocalLLMConfig:
    def test_defaults(self):
        cfg = LocalLLMConfig()
        assert cfg.hf_id == ""
        assert cfg.quant == QuantMode.AUTO
        assert cfg.idle_seconds == 3

    def test_from_env(self):
        env = {
            "LOCAL_LLM_HF_ID": "Qwen/Qwen3-8B-Instruct",
            "LOCAL_LLM_QUANT": "bf16",
            "LOCAL_LLM_IDLE_SEC": "10",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.hf_id == "Qwen/Qwen3-8B-Instruct"
        assert cfg.quant == QuantMode.BF16
        assert cfg.idle_seconds == 10

    def test_from_env_missing_uses_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = LocalLLMConfig.from_env()
        assert cfg.hf_id == ""
        assert cfg.quant == QuantMode.AUTO
        assert cfg.idle_seconds == 3

    def test_to_env_dict(self):
        cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-4B", quant=QuantMode.INT8, idle_seconds=5)
        d = cfg.to_env_dict()
        assert d == {
            "LOCAL_LLM_HF_ID": "Qwen/Qwen3-4B",
            "LOCAL_LLM_QUANT": "8bit",
            "LOCAL_LLM_IDLE_SEC": "5",
            "LOCAL_LLM_GGUF_FILE": "",
            "LOCAL_LLM_CTX_SIZE": "65536",
        }

    def test_defaults_include_no_gguf_file(self):
        cfg = LocalLLMConfig()
        assert cfg.gguf_file is None

    def test_from_env_reads_gguf_file(self):
        env = {"LOCAL_LLM_GGUF_FILE": "model-Q4_K_M.gguf"}
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.gguf_file == "model-Q4_K_M.gguf"

    def test_from_env_treats_empty_gguf_file_as_none(self):
        env = {"LOCAL_LLM_GGUF_FILE": ""}
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.gguf_file is None

    def test_to_env_dict_emits_gguf_file(self):
        cfg = LocalLLMConfig(
            hf_id="anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
            quant=QuantMode.AUTO,
            idle_seconds=3,
            gguf_file="model-Q4_K_M.gguf",
        )
        d = cfg.to_env_dict()
        assert d["LOCAL_LLM_GGUF_FILE"] == "model-Q4_K_M.gguf"

    # ---- n_ctx field (KV cache size knob to control VRAM) ----

    def test_n_ctx_defaults_to_65536(self):
        # Default chosen to fit long script processing comfortably.
        cfg = LocalLLMConfig()
        assert cfg.n_ctx == 65536

    def test_n_ctx_validation_rejects_too_small(self):
        import pytest
        with pytest.raises(Exception):
            LocalLLMConfig(n_ctx=128)  # below ge=512

    def test_n_ctx_validation_rejects_too_large(self):
        import pytest
        with pytest.raises(Exception):
            LocalLLMConfig(n_ctx=999999)  # above le=131072

    def test_from_env_reads_n_ctx(self):
        env = {"LOCAL_LLM_CTX_SIZE": "2048"}
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.n_ctx == 2048

    def test_to_env_dict_emits_n_ctx(self):
        cfg = LocalLLMConfig(n_ctx=8192)
        d = cfg.to_env_dict()
        assert d["LOCAL_LLM_CTX_SIZE"] == "8192"
