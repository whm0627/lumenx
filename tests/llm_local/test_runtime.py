"""Integration tests for LocalRuntime using a tiny real model.

Marked as `slow` because it downloads weights on first run. Skipped when no GPU.
"""
import pytest

# Defer all imports until we know torch + CUDA are available, otherwise
# the entire module fails collection on machines without torch installed.
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False
    torch = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="LocalRuntime requires CUDA GPU",
)

if _HAS_CUDA:
    from src.llm_local.runtime import LocalRuntime, LoadResult  # noqa: E402
    from src.llm_local.vram import QuantMode  # noqa: E402

# A tiny instruct model that's quick to download (~1 GB).
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def runtime():
    rt = LocalRuntime(hf_id=TEST_MODEL, quant=QuantMode.BF16)
    yield rt
    rt.unload()


class TestLocalRuntime:
    def test_load_returns_load_result(self, runtime):
        result = runtime.load()
        assert isinstance(result, LoadResult)
        assert result.hf_id == TEST_MODEL
        assert result.vram_used_mb > 0

    def test_chat_returns_non_empty_string(self, runtime):
        runtime.load()
        out = runtime.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)
        assert isinstance(out, str)
        assert len(out.strip()) > 0

    def test_chat_strips_input_prompt_from_output(self, runtime):
        runtime.load()
        out = runtime.chat(
            [{"role": "user", "content": "Reply with exactly the word: TESTOUTPUT"}],
            max_new_tokens=8,
        )
        # The output must not contain the user's instruction text verbatim
        assert "Reply with exactly" not in out

    def test_unload_frees_vram(self, runtime):
        runtime.load()
        before = torch.cuda.memory_allocated()
        runtime.unload()
        after = torch.cuda.memory_allocated()
        assert after < before
