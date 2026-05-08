"""AudioModelManager — singleton state machine for the local CosyVoice SFT
runtime.

Mirrors ImageModelManager but for TTS. The runtime is much smaller than
the image model (~500MB vs ~13GB) so we don't need GPULock contention
or aggressive offload; we still want explicit state transitions so the
global status footer can show what the local TTS is doing.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.audio_local.manager import AudioModelManager, AudioState


@pytest.fixture(autouse=True)
def _reset_state():
    AudioModelManager.reset()
    yield
    AudioModelManager.reset()


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = AudioModelManager.get()
        b = AudioModelManager.get()
        assert a is b

    def test_reset_creates_fresh(self):
        a = AudioModelManager.get()
        AudioModelManager.reset()
        b = AudioModelManager.get()
        assert a is not b


class TestStatus:
    def test_initial_state_is_unloaded(self):
        s = AudioModelManager.get().status()
        assert s["state"] == AudioState.UNLOADED.value
        assert s["error"] is None

    def test_status_includes_hf_id_and_progress(self):
        s = AudioModelManager.get().status()
        assert "hf_id" in s
        assert "progress" in s


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_transitions_to_ready(self):
        mgr = AudioModelManager.get()
        with patch.object(mgr._inner, "load", return_value=None):
            await mgr.load()
        assert mgr.status()["state"] == AudioState.READY.value

    @pytest.mark.asyncio
    async def test_load_idempotent_when_ready(self):
        mgr = AudioModelManager.get()
        called = {"n": 0}

        def fake_load():
            called["n"] += 1

        with patch.object(mgr._inner, "load", side_effect=fake_load):
            await mgr.load()
            await mgr.load()
        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_load_failure_sets_error_state(self):
        mgr = AudioModelManager.get()
        with patch.object(
            mgr._inner, "load", side_effect=RuntimeError("cosyvoice missing")
        ):
            with pytest.raises(RuntimeError):
                await mgr.load()
        s = mgr.status()
        assert s["state"] == AudioState.ERROR.value
        assert "cosyvoice missing" in s["error"]


class TestSynthesize:
    """The sync entry used by AudioGenerator. Reports SYNTHESIZING during
    the call so the footer shows TTS is busy, not just READY."""

    def test_synthesize_lazy_loads_and_returns_tuple(self, tmp_path):
        mgr = AudioModelManager.get()
        out = str(tmp_path / "out.wav")
        with patch.object(mgr._inner, "load", return_value=None), \
             patch.object(
                 mgr._inner,
                 "synthesize",
                 return_value=(out, 0.0, "local-1"),
             ) as m:
            path, delay, rid = mgr.synthesize("hello", out, voice="中文女")
        m.assert_called_once()
        assert path == out
        assert rid == "local-1"

    def test_synthesize_failure_sets_error_state(self, tmp_path):
        mgr = AudioModelManager.get()
        mgr._loaded = True  # skip lazy load
        with patch.object(
            mgr._inner, "synthesize", side_effect=RuntimeError("cuda OOM")
        ):
            with pytest.raises(RuntimeError):
                mgr.synthesize("x", str(tmp_path / "o.wav"), voice="中文女")
        s = mgr.status()
        assert s["state"] == AudioState.ERROR.value
        assert "cuda OOM" in s["error"]

    def test_synthesize_reports_synthesizing_state_during_call(self, tmp_path):
        mgr = AudioModelManager.get()
        mgr._loaded = True
        observed = []

        def fake_synth(*args, **kwargs):
            observed.append(mgr._state)
            return (str(tmp_path / "o.wav"), 0.0, "rid")

        with patch.object(mgr._inner, "synthesize", side_effect=fake_synth):
            mgr.synthesize("x", str(tmp_path / "o.wav"), voice="中文女")
        assert observed == [AudioState.SYNTHESIZING]
        assert mgr.status()["state"] == AudioState.READY.value


class TestVoiceList:
    """Local provider exposes only CosyVoice-300M-SFT's built-in speaker
    presets — no overlap with cloud DashScope voice IDs."""

    def test_list_voices_returns_cosyvoice_presets(self):
        voices = AudioModelManager.get().list_voices()
        ids = {v["id"] for v in voices}
        assert "中文女" in ids
        assert "中文男" in ids

    def test_list_voices_does_not_include_cloud_voices(self):
        voices = AudioModelManager.get().list_voices()
        ids = {v["id"] for v in voices}
        # Cloud-only Alibaba voices must not appear
        assert not any(vid.startswith("long") for vid in ids)
