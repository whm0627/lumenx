"""Tests for LocalQwenImageModel — local Qwen-Image diffusers wrapper."""
from unittest.mock import MagicMock, patch

import pytest

from src.models.image import ImageGenModel
from src.models.image_local import LocalQwenImageModel
from src.utils.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def _reset_gpu_lock():
    GPULock.reset()
    yield
    GPULock.reset()


def _setup_loaded(tmp_path):
    """Return (instance, fake_pipe, fake_img) with the pipe pre-installed
    so we don't trigger lazy-load (which would import diffusers)."""
    inst = LocalQwenImageModel({})
    fake_pipe = MagicMock()
    fake_img = MagicMock()
    fake_img.save = MagicMock()
    fake_pipe.return_value = MagicMock(images=[fake_img])
    inst._pipe = fake_pipe
    return inst, fake_pipe, fake_img


class TestSubclass:
    def test_is_image_gen_model(self):
        assert issubclass(LocalQwenImageModel, ImageGenModel)


class TestT2I:
    def test_calls_pipe_with_prompt_and_no_image(self, tmp_path):
        inst, pipe, _ = _setup_loaded(tmp_path)
        inst.generate("a cat", str(tmp_path / "o.png"))
        kwargs = pipe.call_args.kwargs
        assert kwargs["prompt"] == "a cat"
        assert "image" not in kwargs

    def test_writes_output_to_provided_path(self, tmp_path):
        inst, _, img = _setup_loaded(tmp_path)
        out = tmp_path / "result.png"
        path, _ = inst.generate("anything", str(out))
        img.save.assert_called_once_with(str(out))
        assert path == str(out)

    def test_returns_duration_seconds(self, tmp_path):
        inst, _, _ = _setup_loaded(tmp_path)
        _, dur = inst.generate("x", str(tmp_path / "o.png"))
        assert isinstance(dur, float)
        assert dur >= 0


class TestSizeParsing:
    def test_parses_wxh_format(self, tmp_path):
        # Asset code uses wanx-style "WIDTH*HEIGHT" strings; we must accept them.
        inst, pipe, _ = _setup_loaded(tmp_path)
        inst.generate("x", str(tmp_path / "o.png"), size="768*1280")
        kwargs = pipe.call_args.kwargs
        assert kwargs["width"] == 768
        assert kwargs["height"] == 1280

    def test_default_size_is_1024_square(self, tmp_path):
        inst, pipe, _ = _setup_loaded(tmp_path)
        inst.generate("x", str(tmp_path / "o.png"))
        kwargs = pipe.call_args.kwargs
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024


class TestI2I:
    def _setup_with_edit(self, tmp_path):
        """Build instance with both T2I + Edit pipes pre-mocked. Refs go
        through the Edit pipe (Qwen-Image-Edit-2509 / QwenImageEditPlusPipeline)
        which is purpose-built for reference-conditioned generation; the
        plain Img2Img pipe just denoises on top and produces 'lightly
        modified copy of input', not actual variants/view changes."""
        inst, t2i_pipe, _ = _setup_loaded(tmp_path)
        edit_pipe = MagicMock()
        fake_img = MagicMock()
        fake_img.save = MagicMock()
        edit_pipe.return_value = MagicMock(images=[fake_img])
        inst._pipe_edit = edit_pipe
        return inst, t2i_pipe, edit_pipe

    def test_single_ref_dispatches_to_edit_pipe(self, tmp_path, monkeypatch):
        inst, t2i_pipe, edit_pipe = self._setup_with_edit(tmp_path)
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"fake")
        loaded = MagicMock()
        opener = MagicMock(return_value=MagicMock(
            convert=MagicMock(return_value=loaded)
        ))
        monkeypatch.setattr("src.models.image_local.Image.open", opener)
        inst.generate(
            "variant of character",
            str(tmp_path / "o.png"),
            ref_image_path=str(ref),
        )
        # T2I pipe MUST NOT be called for ref-conditioned generation.
        t2i_pipe.assert_not_called()
        kwargs = edit_pipe.call_args.kwargs
        assert kwargs["image"] is loaded

    def test_multi_ref_passes_list_to_edit_pipe(self, tmp_path, monkeypatch):
        inst, t2i_pipe, edit_pipe = self._setup_with_edit(tmp_path)
        refs = [tmp_path / f"r{i}.png" for i in range(3)]
        for r in refs:
            r.write_bytes(b"x")
        loaded_imgs = [MagicMock() for _ in refs]
        idx = iter(loaded_imgs)
        opener = MagicMock(side_effect=lambda p: MagicMock(
            convert=MagicMock(return_value=next(idx))
        ))
        monkeypatch.setattr("src.models.image_local.Image.open", opener)
        inst.generate(
            "x",
            str(tmp_path / "o.png"),
            ref_image_paths=[str(r) for r in refs],
        )
        t2i_pipe.assert_not_called()
        kwargs = edit_pipe.call_args.kwargs
        assert kwargs["image"] == loaded_imgs

    def test_no_refs_uses_t2i_pipe(self, tmp_path):
        """Pure T2I (no ref) must NOT touch the edit pipe."""
        inst, t2i_pipe, edit_pipe = self._setup_with_edit(tmp_path)
        inst.generate("a cat", str(tmp_path / "o.png"))
        t2i_pipe.assert_called_once()
        edit_pipe.assert_not_called()


class TestNegativePrompt:
    def test_forwarded_when_provided(self, tmp_path):
        inst, pipe, _ = _setup_loaded(tmp_path)
        inst.generate("x", str(tmp_path / "o.png"), negative_prompt="blurry")
        kwargs = pipe.call_args.kwargs
        assert kwargs.get("negative_prompt") == "blurry"


class TestLazyLoad:
    def test_pipe_not_loaded_at_init(self):
        inst = LocalQwenImageModel({})
        assert inst._pipe is None

    def test_pipe_loaded_on_first_generate_and_reused(self, tmp_path, monkeypatch):
        load_calls = {"n": 0}
        fake_pipe = MagicMock()
        fake_img = MagicMock()
        fake_img.save = MagicMock()
        fake_pipe.return_value = MagicMock(images=[fake_img])

        def fake_load(self):
            load_calls["n"] += 1
            return fake_pipe

        monkeypatch.setattr(LocalQwenImageModel, "_load_pipe", fake_load)
        inst = LocalQwenImageModel({})
        inst.generate("x", str(tmp_path / "o1.png"))
        inst.generate("y", str(tmp_path / "o2.png"))
        assert load_calls["n"] == 1  # loaded once, reused


class TestExclusiveGPULock:
    """LocalQwenImageModel must coexist exclusively with the LLM via GPULock:
    register at init, acquire before loading, release on unload."""

    def test_init_registers_with_gpu_lock(self):
        # Once registered, another runtime calling acquire should evict us.
        inst = LocalQwenImageModel({})
        with patch.object(inst, "unload") as mock_unload:
            # Re-register so the patched method is the one held by the lock
            GPULock.get().register("image", mock_unload)
            GPULock.get().register("llm", MagicMock())
            GPULock.get().acquire("image")  # image becomes current
            GPULock.get().acquire("llm")  # evicts image
            mock_unload.assert_called_once()

    def test_load_pipe_acquires_lock_before_returning(self, monkeypatch):
        """The actual heavy load must claim the GPU before allocating VRAM."""
        inst = LocalQwenImageModel({})
        acquire_log = []
        monkeypatch.setattr(
            "src.utils.gpu_lock.GPULock.acquire",
            lambda self, name: acquire_log.append(name),
        )
        # Stub the diffusers load entirely — we only care lock.acquire happened
        monkeypatch.setattr(
            LocalQwenImageModel, "_construct_pipe", lambda self: MagicMock()
        )
        inst._load_pipe()
        assert "image" in acquire_log

    def test_unload_drops_pipe_reference(self):
        inst = LocalQwenImageModel({})
        inst._pipe = MagicMock()
        inst.unload()
        assert inst._pipe is None

    def test_unload_releases_gpu_lock(self):
        inst = LocalQwenImageModel({})
        GPULock.get().register("image", inst.unload)
        GPULock.get().acquire("image")
        assert GPULock.get().current() == "image"
        inst.unload()
        assert GPULock.get().current() is None

    def test_unload_when_not_loaded_is_noop(self):
        inst = LocalQwenImageModel({})
        # _pipe never set; should not raise
        inst.unload()
        assert inst._pipe is None
