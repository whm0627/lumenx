"""GPULock — single-model-at-a-time enforcement for GPU memory.

Any local runtime (LLM, image, ...) that wants to load weights into VRAM
registers itself, then calls acquire() before its heavy load. Acquire
evicts (calls the unload_fn of) all other registered holders so VRAM is
free for the incoming model.
"""
from unittest.mock import MagicMock

import pytest

from src.utils.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def _reset_singleton():
    GPULock.reset()
    yield
    GPULock.reset()


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = GPULock.get()
        b = GPULock.get()
        assert a is b

    def test_reset_creates_fresh_instance(self):
        a = GPULock.get()
        GPULock.reset()
        b = GPULock.get()
        assert a is not b


class TestRegister:
    def test_register_alone_does_not_evict(self):
        # Registering says "I'm evictable"; it does not assert ownership.
        # Acquire still finds no current holder, so no unload runs.
        lock = GPULock.get()
        llm_unload = MagicMock()
        lock.register("llm", llm_unload)
        lock.register("image", MagicMock())
        lock.acquire("image")
        llm_unload.assert_not_called()

    def test_register_replaces_unload_fn_for_same_name(self):
        # Re-registering must overwrite, so a stale unload_fn never runs.
        lock = GPULock.get()
        first, second = MagicMock(), MagicMock()
        lock.register("llm", first)
        lock.acquire("llm")  # llm is now current
        lock.register("llm", second)  # replace fn
        lock.register("image", MagicMock())
        lock.acquire("image")  # evicts llm
        first.assert_not_called()
        second.assert_called_once()


class TestAcquire:
    def test_acquire_with_no_current_holder_is_noop(self):
        # First load: nothing to evict.
        lock = GPULock.get()
        llm_unload = MagicMock()
        lock.register("llm", llm_unload)
        lock.acquire("llm")
        llm_unload.assert_not_called()

    def test_acquire_evicts_only_the_current_holder(self):
        # Other registered runtimes that haven't loaded must NOT be evicted.
        lock = GPULock.get()
        llm_unload, img_unload, vid_unload = MagicMock(), MagicMock(), MagicMock()
        lock.register("llm", llm_unload)
        lock.register("image", img_unload)
        lock.register("video", vid_unload)
        lock.acquire("llm")  # llm becomes current
        lock.acquire("image")  # evicts llm only
        llm_unload.assert_called_once()
        img_unload.assert_not_called()
        vid_unload.assert_not_called()

    def test_acquire_same_holder_twice_does_not_re_evict(self):
        lock = GPULock.get()
        llm_unload, img_unload = MagicMock(), MagicMock()
        lock.register("llm", llm_unload)
        lock.register("image", img_unload)
        lock.acquire("llm")  # llm now current
        lock.acquire("image")  # evicts llm
        lock.acquire("image")  # noop (already current)
        assert llm_unload.call_count == 1
        img_unload.assert_not_called()

    def test_acquire_a_then_b_then_a_evicts_each_once(self):
        lock = GPULock.get()
        llm_unload, img_unload = MagicMock(), MagicMock()
        lock.register("llm", llm_unload)
        lock.register("image", img_unload)
        lock.acquire("llm")  # llm becomes current (no eviction)
        lock.acquire("image")  # evicts llm
        lock.acquire("llm")  # evicts image
        assert llm_unload.call_count == 1
        assert img_unload.call_count == 1

    def test_unload_exception_still_completes_acquire(self):
        # If the current holder's unload_fn raises, the new holder still
        # becomes current — we don't want to leave the lock in a stuck
        # state because cleanup blew up.
        lock = GPULock.get()
        bad = MagicMock(side_effect=RuntimeError("boom"))
        lock.register("llm", bad)
        lock.register("image", MagicMock())
        lock.acquire("llm")  # llm current
        lock.acquire("image")  # evicts llm (raises) — should still complete
        bad.assert_called_once()
        assert lock.current() == "image"


class TestRelease:
    def test_release_allows_another_acquire_to_evict_self_again(self):
        lock = GPULock.get()
        llm_unload, img_unload = MagicMock(), MagicMock()
        lock.register("llm", llm_unload)
        lock.register("image", img_unload)
        lock.acquire("image")
        lock.release("image")
        # Now image is no longer current — re-acquiring image should evict
        # llm if llm has since acquired
        lock.acquire("llm")
        llm_unload.reset_mock()
        lock.acquire("image")
        llm_unload.assert_called_once()


class TestCurrent:
    def test_current_returns_none_initially(self):
        assert GPULock.get().current() is None

    def test_current_returns_holder_after_acquire(self):
        lock = GPULock.get()
        lock.register("llm", MagicMock())
        lock.acquire("llm")
        assert lock.current() == "llm"

    def test_current_returns_none_after_release(self):
        lock = GPULock.get()
        lock.register("llm", MagicMock())
        lock.acquire("llm")
        lock.release("llm")
        assert lock.current() is None
