"""Tests for LLMAdapter env var handling."""
import os
from unittest.mock import patch

from src.apps.comic_gen.llm_adapter import LLMAdapter


class TestDashscopeModel:
    def test_default_model_falls_back_when_env_unset(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "dashscope"}, clear=False):
            os.environ.pop("DASHSCOPE_MODEL", None)
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "qwen3.5-plus"

    def test_default_model_reads_dashscope_model_env(self):
        env = {"LLM_PROVIDER": "dashscope", "DASHSCOPE_MODEL": "qwen-max"}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "qwen-max"

    def test_openai_path_unchanged(self):
        env = {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4o"}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "gpt-4o"
