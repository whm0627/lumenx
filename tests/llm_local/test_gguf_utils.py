"""Tests for GGUF detection and file priority."""
from unittest.mock import patch

from src.llm_local.gguf_utils import (
    is_gguf_repo,
    pick_default_gguf_file,
    list_gguf_files,
)


class TestIsGgufRepo:
    def test_recognises_gguf_suffix(self):
        assert is_gguf_repo("anthfu/Qwen3.6-35B-A3B-APEX-GGUF") is True

    def test_recognises_lowercase_gguf(self):
        assert is_gguf_repo("bartowski/Qwen2.5-0.5B-Instruct-gguf") is True

    def test_recognises_gguf_anywhere(self):
        assert is_gguf_repo("user/SomeModel-GGUF-Quantized") is True

    def test_rejects_non_gguf(self):
        assert is_gguf_repo("Qwen/Qwen3-8B-Instruct") is False
        assert is_gguf_repo("meta-llama/Llama-3-70B-Instruct") is False


class TestPickDefaultGgufFile:
    def test_picks_q4_k_m_first(self):
        files = ["model-Q5_K_M.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        assert pick_default_gguf_file(files) == "model-Q4_K_M.gguf"

    def test_picks_q5_k_m_when_no_q4_k_m(self):
        files = ["model-Q5_K_M.gguf", "model-Q8_0.gguf"]
        assert pick_default_gguf_file(files) == "model-Q5_K_M.gguf"

    def test_falls_back_to_first_when_no_priority_match(self):
        files = ["model-IQ3_XS.gguf", "model-IQ2_M.gguf"]
        assert pick_default_gguf_file(files) == "model-IQ3_XS.gguf"

    def test_returns_none_for_empty(self):
        assert pick_default_gguf_file([]) is None


class TestListGgufFiles:
    def test_filters_to_gguf_only(self):
        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            mock_api.return_value.list_repo_files.return_value = [
                ".gitattributes",
                "README.md",
                "model-Q4_K_M.gguf",
                "model-Q5_K_M.gguf",
                "config.json",
            ]
            assert list_gguf_files("anthfu/Qwen3.6-35B-A3B-APEX-GGUF") == [
                "model-Q4_K_M.gguf",
                "model-Q5_K_M.gguf",
            ]

    def test_returns_empty_when_no_gguf(self):
        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            mock_api.return_value.list_repo_files.return_value = ["config.json", "model.safetensors"]
            assert list_gguf_files("Qwen/Qwen3-8B-Instruct") == []
