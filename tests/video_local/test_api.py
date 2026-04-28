"""Tests for /video/local/* endpoints."""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.video_local.api import router
from src.video_local.manager import VideoModelManager


@pytest.fixture(autouse=True)
def _reset():
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status_endpoint(client):
    r = client.get("/video/local/status")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body and "quant" in body and "hf_id" in body


def test_load_endpoint_calls_manager_load(client):
    with patch.object(VideoModelManager.get()._inner, "load", return_value=None):
        r = client.post("/video/local/load")
    assert r.status_code == 200
    assert r.json()["state"] == "READY"


def test_load_endpoint_passes_quant(client):
    mgr = VideoModelManager.get()
    with patch.object(mgr._inner, "load", return_value=None), \
         patch.object(mgr._inner, "unload", return_value=None):
        r = client.post("/video/local/load", json={"quant": "Q8_0"})
    assert r.status_code == 200
    assert r.json()["quant"] == "Q8_0"


def test_cancel_endpoint(client):
    mgr = VideoModelManager.get()
    mgr._loaded = True
    with patch.object(mgr._inner, "unload", return_value=None):
        r = client.post("/video/local/cancel")
    assert r.status_code == 200
    assert r.json()["ok"] is True
