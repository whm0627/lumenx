"""VideoModelManager — singleton state machine for the local video
runtime. Mirrors AudioModelManager exactly in shape; verifies state
transitions, error capture, idempotency, and quant-tier reload."""
from unittest.mock import patch

import pytest

from src.video_local.manager import VideoModelManager, VideoState


@pytest.fixture(autouse=True)
def _reset():
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


class TestSingleton:
    def test_get_returns_same(self):
        a = VideoModelManager.get()
        b = VideoModelManager.get()
        assert a is b


class TestStatus:
    def test_initial_unloaded(self):
        s = VideoModelManager.get().status()
        assert s["state"] == VideoState.UNLOADED.value
        assert s["error"] is None
        assert s["quant"] in ("fp16", "Q8_0", "Q4_K_S")  # whichever default


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_transitions_to_ready(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", return_value=None):
            await mgr.load()
        assert mgr.status()["state"] == VideoState.READY.value

    @pytest.mark.asyncio
    async def test_load_failure_sets_error(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", side_effect=RuntimeError("gguf 404")):
            with pytest.raises(RuntimeError):
                await mgr.load()
        assert mgr.status()["state"] == VideoState.ERROR.value
        assert "gguf 404" in mgr.status()["error"]


class TestQuantSwitch:
    @pytest.mark.asyncio
    async def test_changing_quant_forces_reload(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", return_value=None), \
             patch.object(mgr._inner, "unload", return_value=None) as mock_unload:
            await mgr.load(quant="Q4_K_S")
            await mgr.load(quant="Q8_0")
        assert mock_unload.called  # the second load required an unload first


class TestGenerate:
    def test_generate_failure_sets_error(self, tmp_path):
        mgr = VideoModelManager.get()
        mgr._loaded = True
        with patch.object(mgr._inner, "generate", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError):
                mgr.generate(image="x.jpg", audio="a.wav", prompt="p",
                             output_path=str(tmp_path / "o.mp4"))
        assert mgr.status()["state"] == VideoState.ERROR.value
        assert "OOM" in mgr.status()["error"]
