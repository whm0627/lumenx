"""Tests for LlamaCppManager — auto-update of llama.cpp release binary."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.llm_local.llama_cpp_release import LlamaCppManager


def make_install(parent: Path, tag: str) -> Path:
    """Create a fake installed llama.cpp dir with llama-server.exe."""
    d = parent / f"llama.cpp-{tag}"
    d.mkdir(parents=True)
    (d / "llama-server.exe").write_bytes(b"fake-binary")
    return d


class TestInstalledVersions:
    def test_no_installs(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        assert m.installed_versions() == []
        assert m.current_version() is None
        assert m.current_exe_path() is None

    def test_one_install(self, tmp_path):
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        assert m.installed_versions() == ["b8941"]
        assert m.current_version() == "b8941"
        assert m.current_exe_path() == tmp_path / "llama.cpp-b8941" / "llama-server.exe"

    def test_multiple_installs_returns_highest_numbered(self, tmp_path):
        make_install(tmp_path, "b8941")
        make_install(tmp_path, "b9000")
        make_install(tmp_path, "b8500")
        m = LlamaCppManager(runtime_parent=tmp_path)
        # current_version returns the build with the highest numeric component
        assert m.current_version() == "b9000"

    def test_skips_dirs_without_llama_server_exe(self, tmp_path):
        # Half-extracted / corrupt install: dir exists but no exe
        (tmp_path / "llama.cpp-b9000").mkdir()
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        assert m.installed_versions() == ["b8941"]
        assert m.current_version() == "b8941"


class TestLatestRemote:
    def test_returns_tag_from_github_api(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps({"tag_name": "b9001"}).encode()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = lambda *a: None
        with patch("src.llm_local.llama_cpp_release.urllib.request.urlopen",
                   return_value=fake_response):
            assert m.latest_remote() == "b9001"

    def test_returns_none_on_network_failure(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch("src.llm_local.llama_cpp_release.urllib.request.urlopen",
                   side_effect=OSError("network unreachable")):
            assert m.latest_remote() is None

    def test_returns_none_on_malformed_response(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        fake_response = MagicMock()
        fake_response.read.return_value = b"not json"
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = lambda *a: None
        with patch("src.llm_local.llama_cpp_release.urllib.request.urlopen",
                   return_value=fake_response):
            assert m.latest_remote() is None


class TestEnsureLatestAvailable:
    @pytest.mark.asyncio
    async def test_no_op_when_already_latest(self, tmp_path):
        make_install(tmp_path, "b9001")
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value="b9001"), \
             patch.object(m, "_download_and_extract") as mock_dl:
            result = await m.ensure_latest_available()
        mock_dl.assert_not_called()
        assert result == "b9001"

    @pytest.mark.asyncio
    async def test_downloads_when_local_missing(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value="b9001"), \
             patch.object(m, "_download_and_extract") as mock_dl:
            mock_dl.side_effect = lambda tag: make_install(tmp_path, tag)
            result = await m.ensure_latest_available()
        mock_dl.assert_called_once_with("b9001")
        assert result == "b9001"

    @pytest.mark.asyncio
    async def test_downloads_when_local_older(self, tmp_path):
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value="b9001"), \
             patch.object(m, "_download_and_extract") as mock_dl:
            mock_dl.side_effect = lambda tag: make_install(tmp_path, tag)
            result = await m.ensure_latest_available()
        mock_dl.assert_called_once_with("b9001")
        assert result == "b9001"

    @pytest.mark.asyncio
    async def test_falls_back_to_local_when_github_unreachable(self, tmp_path):
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value=None), \
             patch.object(m, "_download_and_extract") as mock_dl:
            result = await m.ensure_latest_available()
        mock_dl.assert_not_called()
        assert result == "b8941"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_local_and_no_remote(self, tmp_path):
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value=None), \
             patch.object(m, "_download_and_extract") as mock_dl:
            result = await m.ensure_latest_available()
        mock_dl.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_keeps_old_version_if_download_fails(self, tmp_path):
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        with patch.object(m, "latest_remote", return_value="b9001"), \
             patch.object(m, "_download_and_extract",
                          side_effect=RuntimeError("HTTP 500")):
            result = await m.ensure_latest_available()
        # Old version still present; current_version still returns it
        assert result == "b8941"
        assert m.current_version() == "b8941"


class TestCleanupOld:
    def test_keeps_two_most_recent(self, tmp_path):
        for tag in ["b8000", "b8500", "b8941", "b9001"]:
            make_install(tmp_path, tag)
        m = LlamaCppManager(runtime_parent=tmp_path)
        deleted = m.cleanup_old(keep=2)
        assert sorted(deleted) == ["b8000", "b8500"]
        assert sorted(m.installed_versions()) == ["b8941", "b9001"]

    def test_no_op_when_count_below_keep(self, tmp_path):
        make_install(tmp_path, "b8941")
        m = LlamaCppManager(runtime_parent=tmp_path)
        assert m.cleanup_old(keep=2) == []
        assert m.installed_versions() == ["b8941"]

    def test_keep_one(self, tmp_path):
        for tag in ["b8000", "b8500", "b8941"]:
            make_install(tmp_path, tag)
        m = LlamaCppManager(runtime_parent=tmp_path)
        deleted = m.cleanup_old(keep=1)
        assert sorted(deleted) == ["b8000", "b8500"]
        assert m.installed_versions() == ["b8941"]
