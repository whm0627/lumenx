"""Tests for /llm/local/* FastAPI router using a stub manager."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.llm_local.api import router
from src.llm_local.manager import ModelManager, ModelState


@pytest.fixture
def app_with_router():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def stub_manager():
    mgr = MagicMock(spec=ModelManager)
    mgr.status.return_value = {
        "state": ModelState.UNLOADED.value,
        "hf_id": "Qwen/Qwen3-8B-Instruct",
        "quant_mode": "auto",
        "vram_used_mb": 0,
        "vram_total_mb": 24576,
        "last_used_ts": 0.0,
        "idle_seconds": 3,
        "error": None,
    }
    mgr.configure = AsyncMock(return_value=None)
    mgr.load = AsyncMock(return_value=mgr.status.return_value)
    mgr.unload = AsyncMock(return_value=None)
    mgr.chat = AsyncMock(return_value="Hello")
    return mgr


@pytest.fixture
def client(app_with_router, stub_manager):
    with patch("src.llm_local.api.ModelManager.get", return_value=stub_manager):
        yield TestClient(app_with_router)


class TestApi:
    def test_runtime_endpoint_reports_install_state(self, client, tmp_path, monkeypatch):
        """GET /llm/local/runtime returns LlamaCppManager state (installed
        versions, current exe path, runtime parent)."""
        from src.llm_local.llama_cpp_release import LlamaCppManager
        # Build a fake install layout
        (tmp_path / "llama.cpp-b8941").mkdir()
        (tmp_path / "llama.cpp-b8941" / "llama-server.exe").write_bytes(b"x")
        fake_mgr = LlamaCppManager(runtime_parent=tmp_path)
        monkeypatch.setattr("src.llm_local.runtime_server._BINARY_MANAGER", fake_mgr)
        resp = client.get("/llm/local/runtime")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_version"] == "b8941"
        assert "b8941" in body["installed_versions"]
        assert body["exe_path"].endswith("llama-server.exe")
        assert str(tmp_path) in body["runtime_parent"]

    def test_runtime_endpoint_reports_empty_when_nothing_installed(self, client, tmp_path, monkeypatch):
        from src.llm_local.llama_cpp_release import LlamaCppManager
        fake_mgr = LlamaCppManager(runtime_parent=tmp_path)
        monkeypatch.setattr("src.llm_local.runtime_server._BINARY_MANAGER", fake_mgr)
        resp = client.get("/llm/local/runtime")
        body = resp.json()
        assert body["current_version"] is None
        assert body["installed_versions"] == []
        assert body["exe_path"] is None

    def test_status(self, client):
        resp = client.get("/llm/local/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "UNLOADED"
        assert body["vram_total_mb"] == 24576

    def test_configure(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 3}
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "persisted": True}
        mock_save.assert_called_once()

    def test_configure_validates_idle_seconds(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 0}
        resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 422  # Pydantic ge=1

    def test_load(self, client):
        resp = client.post("/llm/local/load")
        assert resp.status_code == 200
        assert resp.json()["state"] == "UNLOADED"  # the stub status

    def test_unload(self, client):
        resp = client.post("/llm/local/unload")
        assert resp.status_code == 200
        assert resp.json()["state"] == "UNLOADED"

    def test_test_endpoint(self, client):
        resp = client.post("/llm/local/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["response"] == "Hello"

    def test_cached_lists_models(self, client, tmp_path, monkeypatch):
        # Build a fake HF cache layout
        cache = tmp_path / "models--Qwen--Qwen3-8B-Instruct" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "config.json").write_bytes(b"x" * 1000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "Qwen/Qwen3-8B-Instruct"
        assert body[0]["size_bytes"] == 1000

    def test_configure_accepts_gguf_file(self, client):
        payload = {
            "hf_id": "anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
            "quant": "auto",
            "idle_seconds": 3,
            "gguf_file": "model-Q4_K_M.gguf",
        }
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        # The persisted dict must include LOCAL_LLM_GGUF_FILE
        saved = mock_save.call_args[0][0]
        assert saved["LOCAL_LLM_GGUF_FILE"] == "model-Q4_K_M.gguf"

    def test_configure_omitted_gguf_file_persists_empty(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 3}
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        saved = mock_save.call_args[0][0]
        assert saved["LOCAL_LLM_GGUF_FILE"] == ""

    def test_cached_lists_gguf_files_for_gguf_repo(self, client, tmp_path, monkeypatch, stub_manager):
        # Build a fake HF cache layout with GGUF files
        cache = tmp_path / "models--anthfu--Qwen3.6-35B-A3B-APEX-GGUF" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "model-Q4_K_M.gguf").write_bytes(b"x" * 1000)
        (cache / "model-Q5_K_M.gguf").write_bytes(b"x" * 2000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        # Set the manager's reported active model. spec=ModelManager hides instance
        # attrs, so attach `config` explicitly as a plain MagicMock.
        stub_manager.status.return_value["hf_id"] = "anthfu/Qwen3.6-35B-A3B-APEX-GGUF"
        stub_manager.config = MagicMock(gguf_file="model-Q4_K_M.gguf")
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "anthfu/Qwen3.6-35B-A3B-APEX-GGUF"
        assert sorted(body[0]["gguf_files"]) == ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
        assert body[0]["active_gguf_file"] == "model-Q4_K_M.gguf"

    def test_cached_returns_null_gguf_for_non_gguf_repo(self, client, tmp_path, monkeypatch):
        cache = tmp_path / "models--Qwen--Qwen3-8B-Instruct" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "config.json").write_bytes(b"x" * 1000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert body[0]["gguf_files"] is None
        assert body[0]["active_gguf_file"] is None

    # ---- /cached blob/snapshot size logic (regression for "0 KB during download") ----

    def test_cached_reports_size_from_blobs_during_download(self, client, tmp_path, monkeypatch):
        """During hf-xet download: blobs/ has data, snapshots/ is empty.
        /cached must report blob size, not 0."""
        repo = tmp_path / "models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF"
        (repo / "blobs").mkdir(parents=True)
        (repo / "snapshots").mkdir(parents=True)
        # In-progress: .incomplete file in blobs
        (repo / "blobs" / "abc123.incomplete").write_bytes(b"x" * 5000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
        assert body[0]["size_bytes"] == 5000

    def test_cached_reports_size_from_snapshots_when_blobs_empty(self, client, tmp_path, monkeypatch):
        """After download completes on Windows non-symlink mode: blobs/ may be
        empty and snapshots/<rev> has the actual file. /cached must still
        report the snapshot size."""
        repo = tmp_path / "models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF"
        (repo / "blobs").mkdir(parents=True)  # exists but empty
        snap = repo / "snapshots" / "rev123"
        snap.mkdir(parents=True)
        (snap / "model.gguf").write_bytes(b"x" * 9000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["size_bytes"] == 9000

    def test_cached_takes_max_when_both_blobs_and_snapshots_have_content(self, client, tmp_path, monkeypatch):
        """If both blobs and snapshots have data (symlink case on Linux),
        we take max() to avoid double-counting."""
        repo = tmp_path / "models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF"
        (repo / "blobs").mkdir(parents=True)
        (repo / "blobs" / "abc.bin").write_bytes(b"x" * 7000)
        snap = repo / "snapshots" / "rev123"
        snap.mkdir(parents=True)
        (snap / "model.gguf").write_bytes(b"x" * 7000)  # symlink-equivalent
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert body[0]["size_bytes"] == 7000  # not 14000

    # ---- /cached download_status (downloading / paused / complete) ----

    def test_cached_status_complete_when_snapshot_has_real_file(self, client, tmp_path, monkeypatch):
        """A snapshot with a non-incomplete file = fully downloaded → 'complete'."""
        repo = tmp_path / "models--Qwen--Qwen2.5-0.5B-Instruct-GGUF"
        snap = repo / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
        (repo / "blobs").mkdir()
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert body[0]["download_status"] == "complete"

    def test_cached_status_downloading_when_active_load_matches(self, client, tmp_path, monkeypatch, stub_manager):
        """blobs/.incomplete + manager.state==LOADING + hf_id matches → 'downloading'."""
        repo = tmp_path / "models--anthfu--EvilModel-GGUF"
        (repo / "blobs").mkdir(parents=True)
        (repo / "blobs" / "abc.incomplete").write_bytes(b"x" * 500)
        (repo / "snapshots" / "rev1").mkdir(parents=True)  # empty snapshot
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        # Tell the stub manager: we're LOADING this exact hf_id
        stub_manager.status.return_value = {
            "state": "LOADING",
            "hf_id": "anthfu/EvilModel-GGUF",
            "quant_mode": "auto",
            "vram_used_mb": 0,
            "vram_total_mb": 24576,
            "last_used_ts": 0,
            "idle_seconds": 3,
            "error": None,
        }
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert body[0]["download_status"] == "downloading"

    def test_cached_status_paused_when_manager_idle(self, client, tmp_path, monkeypatch, stub_manager):
        """blobs/.incomplete + manager UNLOADED (or different hf_id) → 'paused'."""
        repo = tmp_path / "models--anthfu--EvilModel-GGUF"
        (repo / "blobs").mkdir(parents=True)
        (repo / "blobs" / "abc.incomplete").write_bytes(b"x" * 500)
        (repo / "snapshots" / "rev1").mkdir(parents=True)  # empty snapshot
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        # Manager is doing nothing OR loading a different model
        stub_manager.status.return_value = {
            "state": "UNLOADED",
            "hf_id": "",
            "quant_mode": "auto",
            "vram_used_mb": 0,
            "vram_total_mb": 24576,
            "last_used_ts": 0,
            "idle_seconds": 3,
            "error": None,
        }
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert body[0]["download_status"] == "paused"

    def test_cached_status_paused_when_loading_different_id(self, client, tmp_path, monkeypatch, stub_manager):
        """Two paused models exist, manager is LOADING repo A; repo B with
        partial bytes shows 'paused' (not 'downloading')."""
        repo_b = tmp_path / "models--otheruser--SomeModel-GGUF"
        (repo_b / "blobs").mkdir(parents=True)
        (repo_b / "blobs" / "xyz.incomplete").write_bytes(b"x" * 500)
        (repo_b / "snapshots" / "rev1").mkdir(parents=True)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        stub_manager.status.return_value = {
            "state": "LOADING",
            "hf_id": "Qwen/SomeOtherRepo-GGUF",  # ≠ repo_b
            "quant_mode": "auto",
            "vram_used_mb": 0,
            "vram_total_mb": 24576,
            "last_used_ts": 0,
            "idle_seconds": 3,
            "error": None,
        }
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert body[0]["download_status"] == "paused"

    def test_cached_skips_empty_repo_dirs(self, client, tmp_path, monkeypatch):
        """A models--xxx dir with no content (empty blobs + empty snapshots)
        must not appear in /cached output."""
        repo = tmp_path / "models--Qwen--EmptyOne"
        (repo / "blobs").mkdir(parents=True)
        (repo / "snapshots").mkdir(parents=True)
        # Also create a real one to make sure we get exactly that one back
        good = tmp_path / "models--Qwen--RealOne"
        (good / "blobs").mkdir(parents=True)
        (good / "blobs" / "f.bin").write_bytes(b"x" * 100)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "Qwen/RealOne"

    # ---- /cancel endpoint ----

    def test_cancel_when_nothing_loading(self, client, stub_manager):
        """When no load is in progress, cancel returns ok with cancelled=False."""
        from src.llm_local.config import LocalLLMConfig
        from src.llm_local.vram import QuantMode
        # spec=ModelManager hides instance attrs; set config explicitly with
        # a real LocalLLMConfig so the endpoint's `LocalLLMConfig(quant=...)`
        # construction validates against the real enum.
        stub_manager.config = LocalLLMConfig(hf_id="", quant=QuantMode.AUTO, idle_seconds=3)
        stub_manager.cancel = AsyncMock(return_value=False)
        stub_manager.configure = AsyncMock(return_value=None)
        with patch("src.llm_local.api.save_user_config"):
            resp = client.post("/llm/local/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["cancelled"] is False

    def test_cancel_keeps_cache_files_on_disk(self, client, tmp_path, monkeypatch, stub_manager):
        """When cancel succeeds, cached files MUST be preserved on disk so the
        user doesn't lose multi-GB downloads. Only manager state + persisted
        hf_id are cleared. Use DELETE /cached for explicit deletion.
        """
        from src.llm_local.config import LocalLLMConfig
        from src.llm_local.vram import QuantMode
        stub_manager.config = LocalLLMConfig(
            hf_id="anthfu/SomeModel-GGUF",
            quant=QuantMode.AUTO,
            idle_seconds=3,
        )
        stub_manager.cancel = AsyncMock(return_value=True)
        stub_manager.configure = AsyncMock(return_value=None)

        # Build a fake cache dir with content (could be partial or complete)
        cache_subdir = tmp_path / "models--anthfu--SomeModel-GGUF"
        (cache_subdir / "blobs").mkdir(parents=True)
        (cache_subdir / "blobs" / "data.bin").write_bytes(b"x" * 1000)
        snap = cache_subdir / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.gguf").write_bytes(b"y" * 100)  # represent: complete file (small for test)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)

        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["cancelled"] is True
        assert body["hf_id"] == "anthfu/SomeModel-GGUF"
        assert body["files_kept"] is True

        # CRITICAL: Cache dir MUST still exist (no auto-delete on cancel)
        assert cache_subdir.exists()
        assert (cache_subdir / "blobs" / "data.bin").exists()
        assert (snap / "model.gguf").exists()

        # save_user_config must have been called with empty hf_id + gguf_file
        mock_save.assert_called_once()
        saved = mock_save.call_args[0][0]
        assert saved["LOCAL_LLM_HF_ID"] == ""
        assert saved["LOCAL_LLM_GGUF_FILE"] == ""

    def test_cancel_with_no_hf_id_does_not_delete_anything(self, client, tmp_path, monkeypatch, stub_manager):
        """If hf_id is empty (somehow), cancel still works but doesn't touch
        any cache directory."""
        from src.llm_local.config import LocalLLMConfig
        from src.llm_local.vram import QuantMode
        stub_manager.config = LocalLLMConfig(hf_id="", quant=QuantMode.AUTO, idle_seconds=3)
        stub_manager.cancel = AsyncMock(return_value=True)
        stub_manager.configure = AsyncMock(return_value=None)

        # Pre-existing unrelated cache dir
        unrelated = tmp_path / "models--Qwen--SomeOther"
        unrelated.mkdir(parents=True)
        (unrelated / "f.bin").write_bytes(b"keep me")
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)

        with patch("src.llm_local.api.save_user_config"):
            resp = client.post("/llm/local/cancel")

        assert resp.status_code == 200
        # Unrelated cache must still exist
        assert unrelated.exists()
        assert (unrelated / "f.bin").read_bytes() == b"keep me"
