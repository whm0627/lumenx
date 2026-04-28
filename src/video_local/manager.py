"""VideoModelManager — singleton state machine wrapping LocalWanS2V.

Parallel to AudioModelManager but for video. Tracks the active quant
tier so the footer can show "VIDEO Wan2.2-S2V Q4_K_S" and switching
tiers triggers an unload + load (no shared weight state).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from enum import Enum
from typing import Any, Dict, Optional

from .wan_s2v_runtime import LocalWanS2V

logger = logging.getLogger(__name__)


class VideoState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"
    LOADING = "LOADING"
    GENERATING = "GENERATING"
    READY = "READY"
    ERROR = "ERROR"


_DEFAULT_QUANT = os.environ.get("LOCAL_VIDEO_QUANT", "Q4_K_S")


class VideoModelManager:
    _instance: Optional["VideoModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[Dict[str, Any]] = None) -> "VideoModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VideoModelManager(config or {})
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._quant = _DEFAULT_QUANT
        self._inner = LocalWanS2V(quant=self._quant)
        self._state = VideoState.UNLOADED
        self._error: Optional[str] = None
        self._loaded = False
        self._gen_progress = 0.0
        self._phase_label = ""

    def status(self) -> Dict[str, Any]:
        progress = (
            self._gen_progress if self._state == VideoState.GENERATING
            else (1.0 if self._state == VideoState.READY else 0.0)
        )
        return {
            "state": self._state.value,
            "quant": self._quant,
            "hf_id": self._inner.hf_id,
            "phase": "",
            "progress": progress,
            "error": self._error,
            "phase_label": self._phase_label,
        }

    async def load(self, quant: Optional[str] = None) -> Dict[str, Any]:
        target = quant or self._quant
        if self._state == VideoState.READY and target == self._quant:
            return self.status()
        # Tier change: unload the existing pipe before constructing a new one
        if self._loaded and target != self._quant:
            await asyncio.to_thread(self._inner.unload)
            self._loaded = False
        self._quant = target
        self._inner.quant = target
        self._state = VideoState.LOADING
        self._error = None
        try:
            await asyncio.to_thread(self._inner.load)
            self._loaded = True
            self._state = VideoState.READY
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.load failed")
            raise
        return self.status()

    async def unload(self) -> None:
        await asyncio.to_thread(self._inner.unload)
        self._state = VideoState.UNLOADED
        self._error = None
        self._loaded = False

    def generate(self, image: str, audio: str, prompt: str, output_path: str, **kwargs) -> str:
        if not self._loaded:
            self._state = VideoState.LOADING
            try:
                self._inner.load()
                self._loaded = True
            except Exception as e:
                self._state = VideoState.ERROR
                self._error = str(e)
                raise
        self._state = VideoState.GENERATING
        self._gen_progress = 0.0
        try:
            result = self._inner.generate(image=image, audio=audio, prompt=prompt,
                                          output_path=output_path, **kwargs)
            self._state = VideoState.READY
            self._gen_progress = 1.0
            self._error = None
            return result
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.generate failed")
            raise
