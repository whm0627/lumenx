"""Tests for ModelManager state machine using a stub runtime."""
import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.llm_local.config import LocalLLMConfig
from src.llm_local.manager import CancelledByUser, ModelManager, ModelState
from src.llm_local.runtime import LoadResult
from src.llm_local.vram import QuantMode


def make_stub_runtime_factory(load_result: LoadResult, chat_response: str = "OK"):
    """Build a factory that returns a fresh MagicMock runtime each call."""
    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = load_result
        rt.chat.return_value = chat_response
        return rt
    return factory


@pytest.fixture
def stub_load_result():
    return LoadResult(
        hf_id="Qwen/Qwen3-8B-Instruct",
        quant_mode=QuantMode.BF16,
        vram_used_mb=15000,
        elapsed_sec=30.0,
    )


@pytest.fixture
def manager(stub_load_result):
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    factory = make_stub_runtime_factory(stub_load_result)
    return ModelManager(config=cfg, runtime_factory=factory)


@pytest.mark.asyncio
async def test_initial_state_is_unloaded(manager):
    status = manager.status()
    assert status["state"] == ModelState.UNLOADED.value


@pytest.mark.asyncio
async def test_chat_lazy_loads(manager):
    out = await manager.chat([{"role": "user", "content": "hi"}])
    assert out == "OK"
    assert manager.status()["state"] == ModelState.READY.value


@pytest.mark.asyncio
async def test_unload_returns_to_unloaded(manager):
    await manager.chat([{"role": "user", "content": "hi"}])
    await manager.unload()
    assert manager.status()["state"] == ModelState.UNLOADED.value


@pytest.mark.asyncio
async def test_chat_with_empty_hf_id_raises(stub_load_result):
    cfg = LocalLLMConfig(hf_id="", quant=QuantMode.AUTO, idle_seconds=3)
    factory = make_stub_runtime_factory(stub_load_result)
    mgr = ModelManager(config=cfg, runtime_factory=factory)
    with pytest.raises(RuntimeError, match="Configure local LLM first"):
        await mgr.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_configure_with_different_id_unloads_current(manager, stub_load_result):
    await manager.chat([{"role": "user", "content": "hi"}])
    assert manager.status()["state"] == ModelState.READY.value
    new_cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-4B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    await manager.configure(new_cfg)
    # Should be unloaded after switching id
    assert manager.status()["state"] == ModelState.UNLOADED.value
    assert manager.status()["hf_id"] == "Qwen/Qwen3-4B-Instruct"


@pytest.mark.asyncio
async def test_idle_watcher_unloads_after_timeout(stub_load_result):
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=1)
    factory = make_stub_runtime_factory(stub_load_result)
    mgr = ModelManager(config=cfg, runtime_factory=factory, watcher_interval_sec=0.2)
    await mgr.start_idle_watcher()
    try:
        await mgr.chat([{"role": "user", "content": "hi"}])
        assert mgr.status()["state"] == ModelState.READY.value
        # Wait long enough for the watcher to trigger
        await asyncio.sleep(2.0)
        assert mgr.status()["state"] == ModelState.UNLOADED.value
    finally:
        await mgr.stop_idle_watcher()


@pytest.mark.asyncio
async def test_concurrent_chat_serialised_by_lock(manager):
    """Two chats started in parallel should both succeed (serialised by lock)."""
    results = await asyncio.gather(
        manager.chat([{"role": "user", "content": "first"}]),
        manager.chat([{"role": "user", "content": "second"}]),
    )
    assert results == ["OK", "OK"]


@pytest.mark.asyncio
async def test_load_oom_steps_down_quant(stub_load_result):
    """OOM at bf16 should auto-step-down to 8bit."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)

    # Track which quant each call used
    call_quants = []

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        call_quants.append(quant)
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        if quant == QuantMode.BF16:
            rt.load.side_effect = RuntimeError("CUDA out of memory")
        else:
            rt.load.return_value = stub_load_result
            rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    out = await mgr.chat([{"role": "user", "content": "hi"}])
    assert out == "OK"
    assert QuantMode.BF16 in call_quants  # tried bf16 first
    assert QuantMode.INT8 in call_quants  # stepped down to 8bit


@pytest.mark.asyncio
async def test_load_non_oom_error_does_not_step_down(stub_load_result):
    """Non-OOM errors propagate immediately, no step-down."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    call_quants = []

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        call_quants.append(quant)
        rt = MagicMock()
        rt.load.side_effect = RuntimeError("model not found")
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    with pytest.raises(RuntimeError, match="model not found"):
        await mgr.chat([{"role": "user", "content": "hi"}])
    assert call_quants == [QuantMode.BF16]  # only tried once


@pytest.mark.asyncio
async def test_factory_dispatches_gguf_repo_to_gguf_runtime(stub_load_result):
    """When hf_id is a GGUF repo, the factory must receive a gguf_file kwarg."""
    cfg = LocalLLMConfig(
        hf_id="anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
        quant=QuantMode.AUTO,
        idle_seconds=3,
        gguf_file="model-Q4_K_M.gguf",
    )
    received_kwargs = {}

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        received_kwargs["hf_id"] = hf_id
        received_kwargs["gguf_file"] = gguf_file
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = stub_load_result
        rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    await mgr.chat([{"role": "user", "content": "hi"}])
    assert received_kwargs["hf_id"] == "anthfu/Qwen3.6-35B-A3B-APEX-GGUF"
    assert received_kwargs["gguf_file"] == "model-Q4_K_M.gguf"


@pytest.mark.asyncio
async def test_factory_receives_n_ctx_from_config(stub_load_result):
    """n_ctx flows from LocalLLMConfig through ModelManager into the factory."""
    cfg = LocalLLMConfig(
        hf_id="anthfu/some-GGUF",
        quant=QuantMode.AUTO,
        idle_seconds=3,
        n_ctx=2048,
    )
    received = {}

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        received["n_ctx"] = n_ctx
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = stub_load_result
        rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    await mgr.chat([{"role": "user", "content": "hi"}])
    assert received["n_ctx"] == 2048


@pytest.mark.asyncio
async def test_factory_does_not_pass_gguf_file_for_non_gguf(stub_load_result):
    """For non-GGUF repos, gguf_file kwarg is None."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    received_kwargs = {}

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        received_kwargs["gguf_file"] = gguf_file
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = stub_load_result
        rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    await mgr.chat([{"role": "user", "content": "hi"}])
    assert received_kwargs["gguf_file"] is None


# ---- Cancel tests ----

@pytest.mark.asyncio
async def test_cancel_returns_false_when_not_loading(stub_load_result):
    """cancel() is a no-op when state is UNLOADED."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    mgr = ModelManager(config=cfg, runtime_factory=make_stub_runtime_factory(stub_load_result))
    result = await mgr.cancel()
    assert result is False
    assert mgr.status()["state"] == ModelState.UNLOADED.value


@pytest.mark.asyncio
async def test_cancel_during_load_transitions_to_unloaded(stub_load_result):
    """cancel() during a blocking load() must inject CancelledByUser, wait for
    the worker thread to exit, then transition state to UNLOADED."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)

    # Stub runtime whose load() blocks on a threading.Event.
    # The async-raise will inject an exception while the worker is in the
    # event.wait() Python-level call, which is exactly when ctypes works.
    block = threading.Event()

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant

        def _blocking_load():
            # Wait up to 30s — but cancel should fire async exception within ~1s
            block.wait(timeout=30)
            return stub_load_result

        rt.load.side_effect = _blocking_load
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)

    # Kick off chat() which triggers lazy load() — runs in executor thread,
    # blocks on the Event. Don't await; we'll cancel it.
    chat_task = asyncio.create_task(mgr.chat([{"role": "user", "content": "hi"}]))

    # Wait for the load to actually start (state transitions to LOADING)
    for _ in range(50):
        if mgr.status()["state"] == ModelState.LOADING.value:
            break
        await asyncio.sleep(0.05)
    assert mgr.status()["state"] == ModelState.LOADING.value, "load() did not enter LOADING"

    # Cancel — should inject CancelledByUser into worker thread
    cancelled = await mgr.cancel()
    assert cancelled is True

    # The chat_task should have raised (CancelledByUser propagates through chat)
    with pytest.raises(BaseException):
        await chat_task

    # State must be UNLOADED
    assert mgr.status()["state"] == ModelState.UNLOADED.value
    # And no error message lingering
    assert mgr.status()["error"] is None

    # Cleanup: release the block in case the thread is still alive somehow
    block.set()


@pytest.mark.asyncio
async def test_cancel_clears_runtime_reference(stub_load_result):
    """After cancel(), the manager's _runtime is None."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)

    block = threading.Event()

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.side_effect = lambda: block.wait(timeout=30) or stub_load_result
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    chat_task = asyncio.create_task(mgr.chat([{"role": "user", "content": "hi"}]))

    for _ in range(50):
        if mgr.status()["state"] == ModelState.LOADING.value:
            break
        await asyncio.sleep(0.05)

    await mgr.cancel()
    with pytest.raises(BaseException):
        await chat_task

    assert mgr._runtime is None
    block.set()


@pytest.mark.asyncio
async def test_cancel_clears_error_state(stub_load_result):
    """If state is ERROR (load failed), cancel resets to UNLOADED and clears
    the error message so the user can recover from a stuck error."""
    cfg = LocalLLMConfig(hf_id="anthfu/EvilModel-GGUF", quant=QuantMode.AUTO, idle_seconds=3)

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.side_effect = RuntimeError("Failed to load model from file: bogus.gguf")
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    # Force a load failure to land in ERROR
    with pytest.raises(RuntimeError, match="Failed to load model"):
        await mgr.chat([{"role": "user", "content": "hi"}])
    assert mgr.status()["state"] == ModelState.ERROR.value
    assert mgr.status()["error"] is not None

    cancelled = await mgr.cancel()
    assert cancelled is True
    assert mgr.status()["state"] == ModelState.UNLOADED.value
    assert mgr.status()["error"] is None


@pytest.mark.asyncio
async def test_cancel_idempotent_when_called_twice(stub_load_result):
    """A second cancel() while already UNLOADED returns False, no state change."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)

    block = threading.Event()

    def factory(hf_id, quant, gguf_file=None, n_ctx=4096):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.side_effect = lambda: block.wait(timeout=30) or stub_load_result
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    chat_task = asyncio.create_task(mgr.chat([{"role": "user", "content": "hi"}]))
    for _ in range(50):
        if mgr.status()["state"] == ModelState.LOADING.value:
            break
        await asyncio.sleep(0.05)

    first = await mgr.cancel()
    with pytest.raises(BaseException):
        await chat_task
    second = await mgr.cancel()

    assert first is True
    assert second is False
    assert mgr.status()["state"] == ModelState.UNLOADED.value
    block.set()
