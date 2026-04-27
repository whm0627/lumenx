"""ImageModelManager — singleton state machine for the local image runtime.

Wraps LocalQwenImageModel so the global status footer can read what's
loaded, what's loading, and what failed without having to peek into
diffusers internals.

State vocabulary mirrors the LLM ModelManager (UNLOADED / LOADING /
DOWNLOADING / READY / ERROR) so the frontend can treat both
identically.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.img_local.manager import ImageModelManager, ImageState
from src.utils.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def _reset_state():
    ImageModelManager.reset()
    GPULock.reset()
    yield
    ImageModelManager.reset()
    GPULock.reset()


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = ImageModelManager.get()
        b = ImageModelManager.get()
        assert a is b

    def test_reset_creates_fresh(self):
        a = ImageModelManager.get()
        ImageModelManager.reset()
        b = ImageModelManager.get()
        assert a is not b


class TestStatus:
    def test_initial_state_is_unloaded(self):
        mgr = ImageModelManager.get()
        s = mgr.status()
        assert s["state"] == ImageState.UNLOADED.value
        assert s["error"] is None

    def test_status_includes_hf_id_and_phase(self):
        mgr = ImageModelManager.get()
        s = mgr.status()
        assert "hf_id" in s
        assert "phase" in s


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_transitions_through_loading_to_ready(self):
        mgr = ImageModelManager.get()
        # Stub the inner pipe construction so load doesn't actually invoke
        # diffusers / download anything
        with patch.object(mgr._inner, "_get_pipe", return_value=MagicMock()):
            await mgr.load()
        assert mgr.status()["state"] == ImageState.READY.value

    @pytest.mark.asyncio
    async def test_load_idempotent_when_already_ready(self):
        mgr = ImageModelManager.get()
        called = {"n": 0}

        def fake_get_pipe():
            called["n"] += 1
            return MagicMock()

        with patch.object(mgr._inner, "_get_pipe", side_effect=fake_get_pipe):
            await mgr.load()
            await mgr.load()  # second call: noop
        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_load_failure_sets_error_state_and_message(self):
        mgr = ImageModelManager.get()
        with patch.object(
            mgr._inner, "_get_pipe", side_effect=RuntimeError("download failed")
        ):
            with pytest.raises(RuntimeError):
                await mgr.load()
        s = mgr.status()
        assert s["state"] == ImageState.ERROR.value
        assert "download failed" in s["error"]


class TestUnload:
    @pytest.mark.asyncio
    async def test_unload_transitions_to_unloaded(self):
        mgr = ImageModelManager.get()
        with patch.object(mgr._inner, "_get_pipe", return_value=MagicMock()):
            await mgr.load()
        with patch.object(mgr._inner, "unload"):
            await mgr.unload()
        assert mgr.status()["state"] == ImageState.UNLOADED.value

    @pytest.mark.asyncio
    async def test_unload_clears_error_state(self):
        mgr = ImageModelManager.get()
        with patch.object(
            mgr._inner, "_get_pipe", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                await mgr.load()
        assert mgr.status()["state"] == ImageState.ERROR.value
        with patch.object(mgr._inner, "unload"):
            await mgr.unload()
        s = mgr.status()
        assert s["state"] == ImageState.UNLOADED.value
        assert s["error"] is None


class TestGenerate:
    """Synchronous entry used by AssetGenerator. Updates state so the
    footer reflects activity even when the user didn't explicitly preload."""

    def test_generate_lazy_loads_and_returns_result(self, tmp_path):
        mgr = ImageModelManager.get()
        out = str(tmp_path / "out.png")
        with patch.object(
            mgr._inner, "generate", return_value=(out, 1.5)
        ) as mock_gen:
            path, dur = mgr.generate("a cat", out)
        mock_gen.assert_called_once()
        assert path == out and dur == 1.5

    def test_generate_failure_sets_error_state(self, tmp_path):
        mgr = ImageModelManager.get()
        with patch.object(
            mgr._inner, "generate", side_effect=RuntimeError("vae OOM")
        ):
            with pytest.raises(RuntimeError):
                mgr.generate("x", str(tmp_path / "o.png"))
        s = mgr.status()
        assert s["state"] == ImageState.ERROR.value
        assert "vae OOM" in s["error"]

    def test_generate_reports_generating_state_during_pipe_call(self, tmp_path):
        """While pipe(prompt=...) is running, state should be GENERATING — not
        LOADING (which means 'loading weights') and not READY (which means
        'idle'). Lets the footer show a distinct 'busy generating' chip."""
        mgr = ImageModelManager.get()
        # Pretend the pipe is already loaded so we skip the LOADING phase.
        mgr._inner._pipe = MagicMock()
        observed_states = []

        def fake_generate(*args, **kwargs):
            observed_states.append(mgr._state)
            return ("/tmp/out.png", 5.0)

        with patch.object(mgr._inner, "generate", side_effect=fake_generate):
            mgr.generate("x", str(tmp_path / "o.png"))

        assert observed_states == [ImageState.GENERATING]
        assert mgr.status()["state"] == ImageState.READY.value

    def test_generate_first_call_loads_then_generates(self, tmp_path):
        """First generate from UNLOADED must traverse LOADING → GENERATING
        → READY so the footer shows the right phase at the right time."""
        mgr = ImageModelManager.get()
        # Simulate unloaded state (pipe is None)
        mgr._inner._pipe = None
        observed = []

        def fake_get_pipe():
            observed.append(("get_pipe", mgr._state))
            mgr._inner._pipe = MagicMock()
            return mgr._inner._pipe

        def fake_generate(*args, **kwargs):
            observed.append(("generate", mgr._state))
            return ("/tmp/out.png", 5.0)

        with patch.object(mgr._inner, "_get_pipe", side_effect=fake_get_pipe), \
             patch.object(mgr._inner, "generate", side_effect=fake_generate):
            mgr.generate("x", str(tmp_path / "o.png"))

        # _get_pipe runs while state=LOADING; pipe() runs while state=GENERATING
        assert observed[0] == ("get_pipe", ImageState.LOADING)
        assert observed[1] == ("generate", ImageState.GENERATING)
        assert mgr.status()["state"] == ImageState.READY.value


class TestImageGenModelInterface:
    """ImageModelManager must satisfy ImageGenModel so AssetGenerator can
    use it interchangeably with WanxImageModel."""

    def test_is_image_gen_model_subclass(self):
        from src.models.image import ImageGenModel
        assert issubclass(ImageModelManager, ImageGenModel)


class TestProgressReporting:
    """status() must report download / load progress so the footer can
    show a percentage instead of an opaque 'LOADING' chip."""

    def test_progress_zero_when_unloaded(self):
        mgr = ImageModelManager.get()
        s = mgr.status()
        assert s["progress"] == 0.0

    def test_progress_uses_cache_size_over_expected(self):
        """While loading, progress = bytes_on_disk / expected_total."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=42):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["progress"] == pytest.approx(0.42)

    def test_progress_caps_below_one_while_loading(self):
        """Even if cache size meets/exceeds expected (e.g. extra files),
        progress shouldn't claim 100% until state actually becomes READY."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=200):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["progress"] < 1.0

    @pytest.mark.asyncio
    async def test_progress_is_one_after_ready(self):
        mgr = ImageModelManager.get()
        with patch.object(mgr._inner, "_get_pipe", return_value=MagicMock()):
            await mgr.load()
        assert mgr.status()["progress"] == 1.0

    def test_progress_zero_when_expected_size_unknown(self):
        """If HF API failed, we have no denominator — report 0 rather than
        some nonsense, footer can fall back to spinner."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=0):
            with patch.object(mgr, "_current_cache_size", return_value=999_999):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["progress"] == 0.0

    def test_phase_is_downloading_when_under_threshold(self):
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=50):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["phase"] == "downloading"

    def test_progress_during_generating_uses_step_callback_value(self):
        """While state==GENERATING, status.progress should reflect the
        diffusers per-step callback, not the cache fill ratio."""
        mgr = ImageModelManager.get()
        mgr._state = ImageState.GENERATING
        mgr._gen_progress = 0.4
        s = mgr.status()
        assert s["state"] == ImageState.GENERATING.value
        assert s["progress"] == pytest.approx(0.4)

    def test_generate_invokes_step_callback_to_update_progress(self, tmp_path):
        """Manager.generate must wire a progress_callback into the inner
        generate so each diffusers step bumps _gen_progress for the footer."""
        mgr = ImageModelManager.get()
        # Simulate already-loaded pipe to skip Phase 1
        mgr._inner._pipe = MagicMock()

        def fake_generate(prompt, output_path, progress_callback=None, **kwargs):
            assert progress_callback is not None, "manager must inject a callback"
            # Simulate diffusers calling the callback after each of 5 steps
            for step in range(1, 6):
                progress_callback(step, 5)
            return (output_path, 1.0)

        with patch.object(mgr._inner, "generate", side_effect=fake_generate):
            mgr.generate("x", str(tmp_path / "o.png"))

        # After completion, state is READY and progress is fully complete.
        assert mgr.status()["state"] == ImageState.READY.value
        assert mgr.status()["progress"] == 1.0

    def test_phase_switches_to_loading_near_completion(self):
        """When cache is mostly populated but state isn't READY yet, weights
        are still being initialised to GPU — surface that as 'loading'."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=97):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["phase"] == "loading"

    def test_state_promotes_to_downloading_when_bytes_arriving(self):
        """While weights are still streaming in, /status state should be
        DOWNLOADING (more informative than the generic LOADING)."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=50):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["state"] == ImageState.DOWNLOADING.value

    def test_state_stays_loading_in_final_init_phase(self):
        """Once cache is mostly full, the diffusers + GPU upload phase
        runs — that's still LOADING, not DOWNLOADING."""
        mgr = ImageModelManager.get()
        with patch.object(mgr, "_fetch_expected_size_bytes", return_value=100):
            with patch.object(mgr, "_current_cache_size", return_value=98):
                mgr._state = ImageState.LOADING
                s = mgr.status()
        assert s["state"] == ImageState.LOADING.value
