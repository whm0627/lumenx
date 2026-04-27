"""Integration tests for GGUFRuntime using a tiny GGUF model.

Skipped when no CUDA GPU; downloads ~400 MB on first run.
"""
import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False
    torch = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="GGUFRuntime is built for CUDA; skipped without GPU",
)

if _HAS_CUDA:
    from src.llm_local.runtime_gguf import GGUFRuntime
    from src.llm_local.runtime import LoadResult

# Tiny instruct GGUF (~400 MB at Q4_K_M) widely available on HF.
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
TEST_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


@pytest.fixture(scope="module")
def runtime():
    rt = GGUFRuntime(hf_id=TEST_MODEL, gguf_file=TEST_FILE)
    yield rt
    rt.unload()


class TestGGUFRuntime:
    def test_load_returns_load_result(self, runtime):
        result = runtime.load()
        assert isinstance(result, LoadResult)
        assert result.hf_id == TEST_MODEL

    def test_chat_returns_non_empty_string(self, runtime):
        runtime.load()
        out = runtime.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)
        assert isinstance(out, str)
        assert len(out.strip()) > 0

    def test_unload_is_idempotent(self, runtime):
        runtime.load()
        runtime.unload()
        runtime.unload()  # second call must not raise
