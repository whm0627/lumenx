"""AudioGenerator must honour TTS_PROVIDER so dialogue synthesis can run
through the local CosyVoice SFT runtime, not always fall back to DashScope.

Mirrors test_storyboard_generator_provider / test_asset_generator_provider:
selection is environment-driven, the local provider routes through a
singleton manager (so multiple AudioGenerator instances share the loaded
CosyVoice weights), and unknown providers are a hard error not a silent
fallback.
"""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _reset():
    # AudioModelManager singleton is reset between tests so each test sees
    # a fresh state machine (mirrors how img_local tests reset their mgr).
    from src.audio_local.manager import AudioModelManager
    AudioModelManager.reset()
    yield
    AudioModelManager.reset()


class TestProviderSelection:
    def test_default_provider_is_dashscope(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        from src.audio.tts import TTSProcessor
        monkeypatch.delenv("TTS_PROVIDER", raising=False)
        # Stub out DashScope SDK init so this test doesn't need credentials
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key-for-test")
        gen = AudioGenerator({})
        assert isinstance(gen.tts, TTSProcessor)

    def test_explicit_dashscope_provider(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        from src.audio.tts import TTSProcessor
        monkeypatch.setenv("TTS_PROVIDER", "dashscope")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key-for-test")
        gen = AudioGenerator({})
        assert isinstance(gen.tts, TTSProcessor)

    def test_local_provider_uses_audio_model_manager(self, monkeypatch):
        # TTS_PROVIDER=local must route through the singleton manager
        # (parallel to ImageModelManager) so the loaded CosyVoice weights
        # are shared across all AudioGenerator instances.
        from src.apps.comic_gen.audio import AudioGenerator
        from src.audio_local.manager import AudioModelManager
        monkeypatch.setenv("TTS_PROVIDER", "local")
        gen = AudioGenerator({})
        assert isinstance(gen.tts, AudioModelManager)

    def test_local_provider_shares_singleton_across_instances(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        monkeypatch.setenv("TTS_PROVIDER", "local")
        a = AudioGenerator({})
        b = AudioGenerator({})
        assert a.tts is b.tts

    def test_provider_value_is_case_insensitive(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        from src.audio_local.manager import AudioModelManager
        monkeypatch.setenv("TTS_PROVIDER", "Local")
        gen = AudioGenerator({})
        assert isinstance(gen.tts, AudioModelManager)

    def test_unknown_provider_raises(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
        with pytest.raises(ValueError, match="elevenlabs"):
            AudioGenerator({})


class TestVoiceListByProvider:
    """Cloud and local CosyVoice expose disjoint voice ID spaces (cloud =
    Alibaba's longxiaochun_v2 etc., local = CosyVoice-300M-SFT's 中文女 / 中文男
    presets). get_available_voices must reflect what the *active* provider
    actually accepts — otherwise the user picks a cloud voice for a local
    project and every dialogue render fails."""

    def test_dashscope_provider_returns_long_voices(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        monkeypatch.setenv("TTS_PROVIDER", "dashscope")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key-for-test")
        gen = AudioGenerator({})
        voices = gen.get_available_voices()
        ids = {v["id"] for v in voices}
        # Cloud voices include the 龙xx series (cloud-only Alibaba IP)
        assert any(vid.startswith("long") for vid in ids)
        # Local presets must NOT leak into the cloud list
        assert "中文女" not in ids

    def test_local_provider_returns_cosyvoice_presets(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        monkeypatch.setenv("TTS_PROVIDER", "local")
        gen = AudioGenerator({})
        voices = gen.get_available_voices()
        ids = {v["id"] for v in voices}
        # CosyVoice-300M-SFT's built-in speaker presets
        assert "中文女" in ids
        assert "中文男" in ids
        # Cloud voice IDs must NOT appear when local is active
        assert not any(vid.startswith("long") for vid in ids)


class TestDialogueOutputFormat:
    def test_local_provider_writes_wav_dialogue(self, monkeypatch):
        from src.apps.comic_gen.audio import AudioGenerator
        from src.apps.comic_gen.models import Character, StoryboardFrame, GenerationStatus

        monkeypatch.setenv("TTS_PROVIDER", "local")
        gen = AudioGenerator({"output_dir": "output/audio"})
        frame = StoryboardFrame(
            id="frame-1",
            scene_id="scene-1",
            dialogue="hello",
        )
        character = Character(
            id="char-1",
            name="Narrator",
            description="test character",
            voice_id="中文男",
        )

        with patch.object(gen.tts, "synthesize", return_value=("unused", 0.0, "rid")) as synth:
            result = gen.generate_dialogue(frame, character)

        output_path = synth.call_args.args[1]
        assert output_path.endswith("frame-1.wav")
        assert result.audio_url.replace("\\", "/").endswith("dialogue/frame-1.wav")
        assert result.status == GenerationStatus.COMPLETED
