"""StoryboardGenerator must honour IMAGE_PROVIDER so storyboard frame
renders go through the same local image runtime as character/asset
generation, not silently fall back to cloud Wanx."""
import pytest

from src.apps.comic_gen.storyboard import StoryboardGenerator
from src.img_local.manager import ImageModelManager
from src.models.image import WanxImageModel
from src.utils.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def _reset():
    ImageModelManager.reset()
    GPULock.reset()
    yield
    ImageModelManager.reset()
    GPULock.reset()


class TestProviderSelection:
    def test_default_provider_is_wanx(self, monkeypatch):
        monkeypatch.delenv("IMAGE_PROVIDER", raising=False)
        gen = StoryboardGenerator({})
        assert isinstance(gen.model, WanxImageModel)

    def test_explicit_wanx_provider(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "wanx")
        gen = StoryboardGenerator({})
        assert isinstance(gen.model, WanxImageModel)

    def test_local_provider_uses_image_model_manager(self, monkeypatch):
        # IMAGE_PROVIDER=local must route through the same singleton manager
        # used by AssetGenerator, so storyboard renders share the loaded
        # pipes and the GPULock state.
        monkeypatch.setenv("IMAGE_PROVIDER", "local")
        gen = StoryboardGenerator({})
        assert isinstance(gen.model, ImageModelManager)

    def test_local_provider_shares_singleton_with_asset_generator(self, monkeypatch):
        # If StoryboardGenerator and AssetGenerator each instantiated their
        # own LocalQwenImageModel, two pipes would coexist in memory and
        # GPULock evictions would race. Both must see the same manager.
        from src.apps.comic_gen.assets import AssetGenerator

        monkeypatch.setenv("IMAGE_PROVIDER", "local")
        story = StoryboardGenerator({})
        asset = AssetGenerator({})
        assert story.model is asset.model

    def test_provider_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "Local")
        gen = StoryboardGenerator({})
        assert isinstance(gen.model, ImageModelManager)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "midjourney")
        with pytest.raises(ValueError, match="midjourney"):
            StoryboardGenerator({})
