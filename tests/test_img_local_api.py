"""Tests for /img/local/* router endpoints."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.img_local.api import router as img_local_router
from src.img_local.manager import ImageModelManager, ImageState
from src.utils.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def _reset():
    ImageModelManager.reset()
    GPULock.reset()
    yield
    ImageModelManager.reset()
    GPULock.reset()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(img_local_router)
    return TestClient(app)


class TestStatus:
    def test_get_status_returns_initial_unloaded(self, client):
        r = client.get("/img/local/status")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == ImageState.UNLOADED.value
        assert body["error"] is None
        assert "hf_id" in body


class TestLoad:
    def test_load_endpoint_triggers_load_and_returns_ready(self, client):
        with patch.object(
            ImageModelManager.get()._inner, "_get_pipe",
            return_value=MagicMock(),
        ):
            r = client.post("/img/local/load")
        assert r.status_code == 200
        assert r.json()["state"] == ImageState.READY.value

    def test_load_failure_returns_500_with_error_in_status(self, client):
        with patch.object(
            ImageModelManager.get()._inner, "_get_pipe",
            side_effect=RuntimeError("hub timeout"),
        ):
            r = client.post("/img/local/load")
        assert r.status_code == 500
        # Subsequent /status should reflect the error
        s = client.get("/img/local/status").json()
        assert s["state"] == ImageState.ERROR.value
        assert "hub timeout" in s["error"]


class TestCancel:
    def test_cancel_when_unloaded_returns_no_op(self, client):
        r = client.post("/img/local/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is False

    def test_cancel_clears_error_state(self, client):
        # Reach ERROR state
        with patch.object(
            ImageModelManager.get()._inner, "_get_pipe",
            side_effect=RuntimeError("boom"),
        ):
            client.post("/img/local/load")
        assert client.get("/img/local/status").json()["state"] == "ERROR"
        # Cancel resets to UNLOADED
        with patch.object(ImageModelManager.get()._inner, "unload"):
            r = client.post("/img/local/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is True
        s = client.get("/img/local/status").json()
        assert s["state"] == "UNLOADED"
        assert s["error"] is None
