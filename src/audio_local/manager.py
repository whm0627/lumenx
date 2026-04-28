"""AudioModelManager — singleton state machine wrapping LocalCosyVoiceTTS.

Parallel to ImageModelManager but for TTS. The runtime is small enough
(~500MB weights, ~2GB VRAM at runtime) that we don't bother with the
GPULock single-model-at-a-time eviction the image manager uses — local
TTS happily coexists with the loaded LLM and Image models on a 24GB card.

State surface mirrors the image manager so the global status footer can
treat both identically:
    UNLOADED → LOADING → READY → SYNTHESIZING (during a call) → READY
                       ↓
                     ERROR (with message captured for the footer)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cosyvoice_runtime import COSYVOICE2_PRESETS, LocalCosyVoiceTTS

logger = logging.getLogger(__name__)


# Same threshold semantics as img_local: bytes-on-disk >= this fraction
# of expected => promote phase from "downloading" to "loading" so the
# user doesn't watch a 99% progress bar that's actually doing GPU init.
DOWNLOAD_DONE_THRESHOLD = 0.95


class AudioState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"
    LOADING = "LOADING"
    SYNTHESIZING = "SYNTHESIZING"
    READY = "READY"
    ERROR = "ERROR"


class AudioModelManager:
    """Singleton wrapper exposing state + sync/async lifecycle for the
    local TTS runtime."""

    _instance: Optional["AudioModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[Dict[str, Any]] = None) -> "AudioModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = AudioModelManager(config or {})
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton. For tests only."""
        with cls._lock:
            cls._instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._inner = LocalCosyVoiceTTS()
        self._state = AudioState.UNLOADED
        self._error: Optional[str] = None
        self._loaded = False  # mirrors _inner._cosyvoice not None, but stays True after a transient error
        self._cached_expected_bytes: int = 0
        self._gen_progress: float = 0.0
        self._phase_label: str = ""

    # ---- progress / cache helpers ---------------------------------------

    def _fetch_expected_size_bytes(self) -> int:
        if self._cached_expected_bytes:
            return self._cached_expected_bytes
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(self._inner.hf_id, files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
            self._cached_expected_bytes = sum(
                getattr(s, "size", None) or 0 for s in siblings
            )
        except Exception:
            logger.exception("AudioModelManager: HfApi.model_info failed")
            self._cached_expected_bytes = 0
        return self._cached_expected_bytes

    def _cache_dir(self) -> Optional[Path]:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            return None
        return Path(hf_home) / "hub" / ("models--" + self._inner.hf_id.replace("/", "--"))

    def _current_cache_size(self) -> int:
        d = self._cache_dir()
        if d is None or not d.exists():
            return 0
        try:
            return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        except Exception:
            return 0

    def _compute_progress(self) -> float:
        if self._state == AudioState.READY:
            return 1.0
        if self._state in (AudioState.UNLOADED, AudioState.ERROR):
            return 0.0
        if self._state == AudioState.SYNTHESIZING:
            return self._gen_progress
        expected = self._fetch_expected_size_bytes()
        if expected <= 0:
            return 0.0
        return min(0.99, self._current_cache_size() / expected)

    def _compute_phase(self, progress: float) -> str:
        if self._state in (
            AudioState.UNLOADED,
            AudioState.READY,
            AudioState.ERROR,
            AudioState.SYNTHESIZING,
        ):
            return ""
        return "loading" if progress >= DOWNLOAD_DONE_THRESHOLD else "downloading"

    # ---- public read API -------------------------------------------------

    def status(self) -> Dict[str, Any]:
        progress = self._compute_progress()
        phase = self._compute_phase(progress)
        state = self._state.value
        # Promote LOADING → DOWNLOADING while bytes are still streaming.
        if self._state == AudioState.LOADING and phase == "downloading":
            state = AudioState.DOWNLOADING.value
        return {
            "state": state,
            "hf_id": self._inner.hf_id,
            "phase": phase,
            "progress": progress,
            "error": self._error,
            "phase_label": self._phase_label or self._inner.active_label,
        }

    def list_voices(self) -> List[Dict[str, str]]:
        """Static preset list — usable before load(). The runtime's
        list_available_spks() can refine this once weights are loaded,
        but for the picker UI the static list is what we want (matches
        the same names the model accepts at inference time)."""
        return list(COSYVOICE2_PRESETS)

    def set_generation_label(self, label: str) -> None:
        self._phase_label = label

    # ---- async lifecycle (used by /audio/local/{load,unload}) ------------

    async def load(self) -> Dict[str, Any]:
        if self._state == AudioState.READY:
            return self.status()
        self._state = AudioState.LOADING
        self._error = None
        try:
            await asyncio.to_thread(self._inner.load)
            self._loaded = True
            self._state = AudioState.READY
        except Exception as e:
            self._state = AudioState.ERROR
            self._error = str(e)
            logger.exception("AudioModelManager.load failed")
            raise
        return self.status()

    async def unload(self) -> None:
        await asyncio.to_thread(self._inner.unload)
        self._state = AudioState.UNLOADED
        self._error = None
        self._loaded = False
        self._cached_expected_bytes = 0
        self._gen_progress = 0.0
        self._phase_label = ""

    # ---- TTSProcessor-compatible sync entry ------------------------------

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
        volume: int = 50,
    ) -> Tuple[str, float, str]:
        """Sync synthesize. Drop-in for src/audio/tts.py:TTSProcessor.synthesize.

        Splits into (LOADING) → SYNTHESIZING → READY so the footer shows
        what's actually happening, even if the user didn't explicitly
        preload via /audio/local/load."""
        if not self._loaded:
            self._state = AudioState.LOADING
            self._error = None
            try:
                self._inner.load()
                self._loaded = True
            except Exception as e:
                self._state = AudioState.ERROR
                self._error = str(e)
                logger.exception("AudioModelManager.synthesize: load failed")
                raise

        self._state = AudioState.SYNTHESIZING
        self._gen_progress = 0.0

        def _step_progress(step: int, total: int) -> None:
            self._gen_progress = step / total if total else 0.0

        try:
            result = self._inner.synthesize(
                text,
                output_path,
                voice=voice,
                speech_rate=speech_rate,
                pitch_rate=pitch_rate,
                volume=volume,
                progress_callback=_step_progress,
            )
            self._state = AudioState.READY
            self._gen_progress = 1.0
            self._error = None
            return result
        except Exception as e:
            self._state = AudioState.ERROR
            self._error = str(e)
            logger.exception("AudioModelManager.synthesize: pipe call failed")
            raise

    # ---- TTSProcessor static-method parity -------------------------------

    @staticmethod
    def list_voices_static() -> Dict[str, Dict[str, str]]:
        """Mirror of TTSProcessor.list_voices() shape — keyed by id, with
        metadata. Some callers use the static form."""
        return {v["id"]: v for v in COSYVOICE2_PRESETS}
