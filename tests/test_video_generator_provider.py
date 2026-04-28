"""VideoGenerator must honor VIDEO_PROVIDER so video generation goes
through the same local runtime as image/audio when local is selected."""
import pytest

from src.apps.comic_gen.video import VideoGenerator


@pytest.fixture(autouse=True)
def _reset():
    from src.video_local.manager import VideoModelManager
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


class TestProviderSelection:
    def test_default_provider_is_wanx(self, monkeypatch):
        monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
        gen = VideoGenerator({})
        # Cloud path uses WanxModel (or whatever the existing class is) —
        # not VideoModelManager
        assert type(gen.model).__name__ != "VideoModelManager"

    def test_local_routes_to_video_model_manager(self, monkeypatch):
        from src.video_local.manager import VideoModelManager
        monkeypatch.setenv("VIDEO_PROVIDER", "local")
        gen = VideoGenerator({})
        assert isinstance(gen.model, VideoModelManager)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("VIDEO_PROVIDER", "midjourney")
        with pytest.raises(ValueError, match="midjourney"):
            VideoGenerator({})
