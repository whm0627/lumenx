"""GPULock — single-model-at-a-time enforcement for GPU memory.

The premise: on a 24GB consumer GPU we can't fit a 35B LLM and a 20B
image model simultaneously. Loading any local runtime is therefore
exclusive — before a runtime loads weights, all other registered
runtimes must release their VRAM.

Usage:
    # At runtime construction
    GPULock.get().register("llm", self.unload_sync)

    # Right before doing the heavy load() that allocates VRAM
    GPULock.get().acquire("llm")

    # When idle-unloading or explicit unload completes
    GPULock.get().release("llm")
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class GPULock:
    """Process-wide singleton coordinating exclusive VRAM ownership."""

    _instance: Optional["GPULock"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.RLock()
        self._unload_fns: Dict[str, Callable[[], None]] = {}
        self._current: Optional[str] = None

    @classmethod
    def get(cls) -> "GPULock":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = GPULock()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton — for tests only."""
        with cls._instance_lock:
            cls._instance = None

    def register(self, name: str, unload_fn: Callable[[], None]) -> None:
        """Make `name`'s runtime evictable. Re-registering replaces the fn."""
        with self._lock:
            self._unload_fns[name] = unload_fn

    def acquire(self, name: str) -> None:
        """Mark `name` as the GPU's sole holder, evicting the prior holder.

        Idempotent: re-acquiring while already current does nothing.
        Only the *current* holder is evicted — other registered runtimes
        that haven't loaded weights are left alone (their unload would
        be a wasteful no-op). If the prior holder's unload raises, we
        still hand ownership to `name` rather than wedging the lock.
        """
        with self._lock:
            if self._current == name:
                return
            prior = self._current
            self._current = name  # set first so an unload raise doesn't wedge
            if prior is not None and prior in self._unload_fns:
                logger.info(f"GPULock: evicting {prior} for {name}")
                try:
                    self._unload_fns[prior]()
                except Exception:
                    logger.exception(
                        f"GPULock: unload_fn for {prior} raised; "
                        f"{name} still acquired"
                    )

    def release(self, name: str) -> None:
        """If `name` is the current holder, mark the GPU as free."""
        with self._lock:
            if self._current == name:
                self._current = None

    def current(self) -> Optional[str]:
        with self._lock:
            return self._current
