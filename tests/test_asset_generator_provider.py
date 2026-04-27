"""AssetGenerator must honour IMAGE_PROVIDER env to swap between Wanx
(cloud) and the local image runtime. Local routes through
ImageModelManager so the footer can see the same state."""
import pytest

from src.apps.comic_gen.assets import AssetGenerator
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
        gen = AssetGenerator({})
        assert isinstance(gen.model, WanxImageModel)

    def test_explicit_wanx_provider(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "wanx")
        gen = AssetGenerator({})
        assert isinstance(gen.model, WanxImageModel)

    def test_local_provider_uses_image_model_manager(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "local")
        gen = AssetGenerator({})
        # Manager is the public face — its state is what the footer reads.
        assert isinstance(gen.model, ImageModelManager)

    def test_local_provider_is_singleton_across_instances(self, monkeypatch):
        # Two AssetGenerators in the same process must share the same manager
        # so a load triggered by one is visible to the other (and the footer).
        monkeypatch.setenv("IMAGE_PROVIDER", "local")
        a = AssetGenerator({})
        b = AssetGenerator({})
        assert a.model is b.model

    def test_provider_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "Local")
        gen = AssetGenerator({})
        assert isinstance(gen.model, ImageModelManager)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "midjourney")
        with pytest.raises(ValueError, match="midjourney"):
            AssetGenerator({})
