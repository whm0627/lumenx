"""Tests for GGUF detection and file priority."""
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm_local.gguf_utils import (
    is_gguf_repo,
    pick_default_gguf_file,
    list_gguf_files,
)


@pytest.fixture(autouse=True)
def _isolated_hf_home(tmp_path, monkeypatch):
    """Point HF_HOME at an empty tmpdir so list_gguf_files's local-first
    behaviour doesn't accidentally pick up the dev machine's real cache
    when a test means to exercise the HF-API code path."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home_isolated"))


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


class TestListGgufFilesLocalFirst:
    """Local cache always wins: if the model's already on disk we use that
    without touching the network. HF API is only consulted to discover
    files for a model that isn't cached yet."""

    def test_uses_local_cache_without_calling_hf_api(self, tmp_path, monkeypatch):
        # Cache has the gguf — HF API must NOT be called.
        hf_home = tmp_path / "hf"
        snap = hf_home / "hub" / "models--anthfu--Qwen3.6-35B-A3B-APEX-GGUF" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "Qwen3.6-35B-A3B-APEX-I-Mini.gguf").write_bytes(b"fake gguf")
        monkeypatch.setenv("HF_HOME", str(hf_home))

        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            files = list_gguf_files("anthfu/Qwen3.6-35B-A3B-APEX-GGUF")
        assert files == ["Qwen3.6-35B-A3B-APEX-I-Mini.gguf"]
        mock_api.assert_not_called()

    def test_falls_through_to_hf_api_when_cache_empty(self, tmp_path, monkeypatch):
        # Nothing cached locally — go to HF API to discover what to download.
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            mock_api.return_value.list_repo_files.return_value = [
                "config.json", "model-Q4_K_M.gguf", "tokenizer.json",
            ]
            files = list_gguf_files("owner/fresh-repo-GGUF")
        assert files == ["model-Q4_K_M.gguf"]

    def test_uses_latest_snapshot_when_multiple_revs_cached(self, tmp_path, monkeypatch):
        # If multiple snapshot revs are present (e.g. user updated the model),
        # the most recent one is the one to scan.
        import time as _t
        hf_home = tmp_path / "hf"
        repo = hf_home / "hub" / "models--owner--repo-GGUF" / "snapshots"
        old = repo / "old_rev"; old.mkdir(parents=True)
        (old / "old.gguf").write_bytes(b"x")
        _t.sleep(0.05)
        new = repo / "new_rev"; new.mkdir()
        (new / "new.gguf").write_bytes(b"x")
        monkeypatch.setenv("HF_HOME", str(hf_home))

        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            files = list_gguf_files("owner/repo-GGUF")
        assert files == ["new.gguf"]
        mock_api.assert_not_called()
