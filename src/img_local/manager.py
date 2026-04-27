"""ImageModelManager — singleton state machine wrapping LocalQwenImageModel.

Purpose: give the global status footer + /img/local/* endpoints a single
read-only view of "what's the local image runtime doing right now",
parallel to LLM ModelManager. The actual diffusers + GPULock work
still lives in LocalQwenImageModel; this layer adds:

- explicit state transitions (UNLOADED → LOADING → READY / ERROR)
- async load/unload entry points so the API endpoints can dispatch
  heavy work to a thread executor without blocking the event loop
- error capture so the footer can surface the failure message
- ImageGenModel subclass conformance so AssetGenerator can use it
  in place of WanxImageModel
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..models.image import ImageGenModel
from ..models.image_local import LocalQwenImageModel

logger = logging.getLogger(__name__)

# Treat the cache as "loading rather than downloading" once this fraction
# of the expected weights have arrived — diffusers init + GPU upload is
# fast (~30s) but doesn't grow the disk cache, so we'd otherwise show a
# stalled 99% bar.
DOWNLOAD_DONE_THRESHOLD = 0.95


class ImageState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"  # bytes streaming from HF to local cache
    LOADING = "LOADING"          # cache → GPU + torchao quant
    GENERATING = "GENERATING"    # pipe(prompt=...) is running
    READY = "READY"              # idle, weights on GPU
    ERROR = "ERROR"


class ImageModelManager(ImageGenModel):
    """Singleton wrapper exposing state + async lifecycle for the local image
    runtime."""

    _instance: Optional["ImageModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[Dict[str, Any]] = None) -> "ImageModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ImageModelManager(config or {})
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton. For tests only."""
        with cls._lock:
            cls._instance = None

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._inner = LocalQwenImageModel(config)
        self._state = ImageState.UNLOADED
        self._error: Optional[str] = None
        # Cached HF API result per-repo (T2I and Edit have different sizes,
        # we lazily fetch each when first observed). Re-built on unload.
        self._cached_expected_bytes: Dict[str, int] = {}
        # Step-level progress while state == GENERATING (0..1). Updated by a
        # diffusers callback_on_step_end hook installed in generate().
        self._gen_progress: float = 0.0
        # Optional human-readable label set by the caller (e.g.
        # AssetGenerator passes "full_body" / "three_view" / "headshot")
        # so the footer can show *which* phase of an "all"-style multi-call
        # batch is currently running, not just yet-another 0..100%.
        self._phase_label: str = ""

    # ---- progress helpers -----------------------------------------------

    def _active_repo_id(self) -> str:
        """Repo whose cache we should be measuring right now. While the
        Edit pipe is loading, that's edit_hf_id; otherwise T2I hf_id."""
        return getattr(self._inner, "active_hf_id", None) or self._inner.hf_id

    def _fetch_expected_size_bytes(self) -> int:
        """Total bytes we expect to download for the *currently-loading*
        repo. Cached per-repo so flipping between T2I and Edit doesn't
        reuse the wrong denominator."""
        repo = self._active_repo_id()
        if not isinstance(self._cached_expected_bytes, dict):
            self._cached_expected_bytes = {}
        if repo in self._cached_expected_bytes:
            return self._cached_expected_bytes[repo]
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo, files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
            total = sum(
                getattr(s, "size", None) or 0 for s in siblings
            )
            self._cached_expected_bytes[repo] = total
            return total
        except Exception:
            logger.exception("ImageModelManager: HfApi.model_info failed")
            self._cached_expected_bytes[repo] = 0
            return 0

    def _cache_dir(self) -> Optional[Path]:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            return None
        repo_dir = "models--" + self._active_repo_id().replace("/", "--")
        return Path(hf_home) / "hub" / repo_dir

    def _current_cache_size(self) -> int:
        d = self._cache_dir()
        if d is None or not d.exists():
            return 0
        try:
            return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        except Exception:
            return 0

    def _compute_progress(self) -> float:
        if self._state == ImageState.READY:
            return 1.0
        if self._state in (ImageState.UNLOADED, ImageState.ERROR):
            return 0.0
        if self._state == ImageState.GENERATING:
            # Step-level progress comes from the diffusers callback in
            # generate(); fall back to 0 if the first step hasn't fired yet.
            return self._gen_progress
        # LOADING / DOWNLOADING — derive from cache fill.
        expected = self._fetch_expected_size_bytes()
        if expected <= 0:
            return 0.0
        current = self._current_cache_size()
        # Cap at 0.99 — only state=READY hits 1.0.
        return min(0.99, current / expected)

    def _compute_phase(self, progress: float) -> str:
        if self._state in (
            ImageState.UNLOADED,
            ImageState.READY,
            ImageState.ERROR,
            ImageState.GENERATING,
        ):
            return ""
        return "loading" if progress >= DOWNLOAD_DONE_THRESHOLD else "downloading"

    # ---- public read API -------------------------------------------------

    def status(self) -> Dict[str, Any]:
        progress = self._compute_progress()
        phase = self._compute_phase(progress)
        # Promote the "downloading" sub-phase to the user-facing state so
        # the footer pill says DOWNLOADING (not the generic LOADING) while
        # bytes are still arriving.
        state = self._state.value
        if self._state == ImageState.LOADING and phase == "downloading":
            state = ImageState.DOWNLOADING.value
        # Report whichever pipe is currently in use (T2I default vs Edit
        # for ref-conditioned), so the footer shows the model the user is
        # actually waiting on rather than a misleading constant.
        active_hf_id = getattr(self._inner, "active_hf_id", None) or self._inner.hf_id

        # Per-pipe view so the footer can show both T2I and Edit models
        # (each with their own load/active state) instead of just one.
        active_states = {
            ImageState.LOADING.value,
            ImageState.DOWNLOADING.value,
            ImageState.GENERATING.value,
        }
        pipes: list = []
        if hasattr(self._inner, "_pipe"):
            t2i_id = getattr(self._inner, "hf_id", "")
            t2i_loaded = self._inner._pipe is not None
            t2i_active = (active_hf_id == t2i_id) and (state in active_states)
            pipes.append({
                "role": "t2i",
                "hf_id": t2i_id,
                "loaded": t2i_loaded,
                "state": state if t2i_active else ("READY" if t2i_loaded else "UNLOADED"),
                "progress": progress if t2i_active else (1.0 if t2i_loaded else 0.0),
            })
        if hasattr(self._inner, "edit_hf_id"):
            edit_id = self._inner.edit_hf_id
            edit_loaded = getattr(self._inner, "_pipe_edit", None) is not None
            edit_active = (active_hf_id == edit_id) and (state in active_states)
            pipes.append({
                "role": "edit",
                "hf_id": edit_id,
                "loaded": edit_loaded,
                "state": state if edit_active else ("READY" if edit_loaded else "UNLOADED"),
                "progress": progress if edit_active else (1.0 if edit_loaded else 0.0),
            })

        return {
            "state": state,
            "hf_id": active_hf_id,
            "phase": phase,
            "progress": progress,
            "error": self._error,
            "phase_label": self._phase_label,
            "pipes": pipes,
        }

    def set_generation_label(self, label: str) -> None:
        """Caller-facing label for the in-flight generation (e.g. asset
        type, variant index). Surfaced via status() so the UI can show
        'Generating full_body 35%' instead of yet another nameless 0..100%."""
        self._phase_label = label

    # ---- async lifecycle (used by /img/local/{load,unload}) -------------

    async def load(self) -> Dict[str, Any]:
        if self._state == ImageState.READY:
            return self.status()
        self._state = ImageState.LOADING
        self._error = None
        try:
            await asyncio.to_thread(self._inner._get_pipe)
            self._state = ImageState.READY
        except Exception as e:
            self._state = ImageState.ERROR
            self._error = str(e)
            logger.exception("ImageModelManager.load failed")
            raise
        return self.status()

    async def unload(self) -> None:
        await asyncio.to_thread(self._inner.unload)
        self._state = ImageState.UNLOADED
        self._error = None
        self._cached_expected_bytes = {}  # let next load re-query if hf_id changed
        self._gen_progress = 0.0
        self._phase_label = ""

    # ---- ImageGenModel sync entry (used by AssetGenerator) --------------

    def generate(
        self,
        prompt: str,
        output_path: str,
        **kwargs: Any,
    ) -> Tuple[str, float]:
        """Sync generate. Splits into two state phases:
        - LOADING / DOWNLOADING while pipe is being constructed
        - GENERATING while pipe(...) is actively producing the image
        Both lazy-load on first call."""
        # Phase 1: ensure the *correct* pipe (T2I vs Edit) is loaded
        # BEFORE entering GENERATING — otherwise a fresh Edit pipe load
        # (~17GB download + init) happens silently inside inner.generate
        # while status reports GENERATING with 0% for many minutes.
        needs_edit = bool(kwargs.get("ref_image_path")) or bool(kwargs.get("ref_image_paths"))
        needed_pipe_loaded = (
            self._inner._pipe_edit is not None if needs_edit
            else self._inner._pipe is not None
        )
        if not needed_pipe_loaded:
            self._state = ImageState.LOADING
            self._error = None
            try:
                if needs_edit:
                    self._inner._get_edit_pipe()
                else:
                    self._inner._get_pipe()
            except Exception as e:
                self._state = ImageState.ERROR
                self._error = str(e)
                logger.exception("ImageModelManager.generate: load failed")
                raise

        # Phase 2: actual pipe call.
        self._state = ImageState.GENERATING
        self._gen_progress = 0.0

        def _step_progress(step: int, total: int) -> None:
            self._gen_progress = step / total if total else 0.0

        # Don't override an explicit caller-provided progress_callback; chain
        # ours so both the manager's UI tracking and the caller's logic run.
        caller_cb = kwargs.get("progress_callback")
        if caller_cb is None:
            kwargs["progress_callback"] = _step_progress
        else:
            def _chained(step: int, total: int) -> None:
                _step_progress(step, total)
                try:
                    caller_cb(step, total)
                except Exception:
                    logger.exception("caller progress_callback raised")
            kwargs["progress_callback"] = _chained

        try:
            result = self._inner.generate(prompt, output_path, **kwargs)
            self._state = ImageState.READY
            self._gen_progress = 1.0
            self._error = None  # success clears any stale error from a prior failure
            return result
        except Exception as e:
            self._state = ImageState.ERROR
            self._error = str(e)
            logger.exception("ImageModelManager.generate: pipe call failed")
            raise
