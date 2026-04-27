"""FastAPI router exposing /llm/local/* endpoints."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .config import LocalLLMConfig
from .manager import ModelManager
from .vram import QuantMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm/local", tags=["local-llm"])

# Resolved HF cache directory (set in __init__.py via HF_HOME).
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


# ---- request / response models ----

class ConfigurePayload(BaseModel):
    hf_id: str = Field(..., min_length=1)
    quant: QuantMode = QuantMode.AUTO
    idle_seconds: int = Field(default=3, ge=1, le=3600)
    gguf_file: Optional[str] = Field(default=None)
    n_ctx: int = Field(default=65536, ge=512, le=131072,
                       description="Context window size; lower = less VRAM")


class CachedModelInfo(BaseModel):
    hf_id: str
    size_bytes: int
    snapshot_path: str
    gguf_files: Optional[List[str]] = None
    active_gguf_file: Optional[str] = None
    download_status: str = "complete"  # "downloading" | "paused" | "complete"
    expected_size_bytes: Optional[int] = None


# ---- save_user_config import (lazy to avoid circular dep at import time) ----

def save_user_config(config_dict: dict) -> None:
    """Thin re-export so tests can monkeypatch a single name in this module."""
    from src.apps.comic_gen.api import save_user_config as _impl
    _impl(config_dict)


# ---- endpoints ----

@router.get("/runtime")
def get_runtime_info() -> dict:
    """Diagnostic: report installed llama.cpp release info."""
    from .runtime_server import get_binary_manager
    mgr = get_binary_manager()
    current = mgr.current_version()
    exe_path = mgr.current_exe_path()
    return {
        "current_version": current,
        "installed_versions": sorted(mgr.installed_versions(), reverse=True),
        "exe_path": str(exe_path) if exe_path else None,
        "runtime_parent": str(mgr.runtime_parent),
    }


@router.get("/status")
def get_status() -> dict:
    return ModelManager.get().status()


@router.post("/configure")
async def configure(payload: ConfigurePayload) -> dict:
    cfg = LocalLLMConfig(
        hf_id=payload.hf_id,
        quant=payload.quant,
        idle_seconds=payload.idle_seconds,
        gguf_file=payload.gguf_file,
        n_ctx=payload.n_ctx,
    )
    save_user_config({"LLM_PROVIDER": "local", **cfg.to_env_dict()})
    # Reflect into process env so subsequent reads see new values immediately
    os.environ["LLM_PROVIDER"] = "local"
    for k, v in cfg.to_env_dict().items():
        os.environ[k] = v
    await ModelManager.get().configure(cfg)
    return {"ok": True, "persisted": True}


@router.post("/load")
async def load() -> dict:
    try:
        return await ModelManager.get().load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload() -> dict:
    await ModelManager.get().unload()
    return ModelManager.get().status()


@router.post("/cancel")
async def cancel_load() -> dict:
    """Cancel an in-flight download/load OR dismiss an ERROR state.

    Stops the active worker thread (if any), resets manager state to UNLOADED,
    and clears LOCAL_LLM_HF_ID + LOCAL_LLM_GGUF_FILE from persisted config.

    Does NOT delete cached files on disk — partial blobs and completed
    snapshots are preserved so the user can resume later or keep a
    fully-downloaded-but-can't-load model around (e.g. waiting for llama.cpp
    to add architecture support). Use DELETE /llm/local/cached?hf_id=... to
    explicitly free disk space.
    """
    mgr = ModelManager.get()
    hf_id = mgr.config.hf_id
    cancelled = await mgr.cancel()

    if not cancelled:
        return {"ok": True, "cancelled": False, "message": "No load was in progress."}

    # Clear persisted state so next session doesn't try to re-load this id
    save_user_config({"LOCAL_LLM_HF_ID": "", "LOCAL_LLM_GGUF_FILE": ""})
    os.environ["LOCAL_LLM_HF_ID"] = ""
    os.environ["LOCAL_LLM_GGUF_FILE"] = ""
    # Also reset the manager's in-memory config so /status reflects the change
    from .config import LocalLLMConfig
    await mgr.configure(LocalLLMConfig(
        hf_id="",
        quant=mgr.config.quant,
        idle_seconds=mgr.config.idle_seconds,
        gguf_file=None,
    ))

    return {"ok": True, "cancelled": True, "hf_id": hf_id, "files_kept": True}


@router.post("/test")
async def test_chat() -> dict:
    t0 = time.monotonic()
    try:
        response = await ModelManager.get().chat(
            [{"role": "user", "content": "Say hello in one word."}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok": bool(response.strip()),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "response": response,
    }


@router.get("/cached")
def list_cached() -> List[CachedModelInfo]:
    """Scan HF cache layout `models--<org>--<name>/snapshots/<rev>` and report sizes.

    For GGUF repos, also reports the list of .gguf files present in the latest
    snapshot and which one is currently active (per ModelManager config).
    """
    if not HF_CACHE_DIR.exists():
        return []

    from .gguf_utils import is_gguf_repo

    # Read active hf_id + manager state in one shot so download_status can
    # distinguish "downloading" (we're actively loading this id) from "paused"
    # (partial bytes on disk but no active worker).
    # NOTE: separate try blocks because manager.config is an instance attr
    # that MagicMock(spec=ModelManager) hides — accessing it in tests raises
    # AttributeError which would mask the manager status read.
    try:
        mgr_status = ModelManager.get().status()
        active_hf_id = mgr_status.get("hf_id", "")
        active_state = mgr_status.get("state", "UNLOADED")
    except Exception:
        active_hf_id = ""
        active_state = "UNLOADED"
    try:
        active_gguf = getattr(ModelManager.get().config, "gguf_file", None)
    except Exception:
        active_gguf = None

    out: List[CachedModelInfo] = []
    for entry in HF_CACHE_DIR.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        # 'models--Qwen--Qwen3-8B-Instruct' -> 'Qwen/Qwen3-8B-Instruct'
        parts = entry.name[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        hf_id = "/".join(parts)

        # During hf-xet download, files land in blobs/<hash>(.incomplete) first
        # and only get linked into snapshots/<rev> on completion. Sum whichever
        # has content so the UI shows real-time progress without going back to
        # "0 KB" right after a download finishes (blobs is emptied at that
        # point on Windows non-symlink mode).
        blobs_dir = entry / "blobs"
        snapshots_dir = entry / "snapshots"
        blobs_size = 0
        snapshot_size = 0
        if blobs_dir.exists():
            blobs_size = sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file())
        if snapshots_dir.exists():
            snaps = list(snapshots_dir.iterdir())
            if snaps:
                latest = max(snaps, key=lambda p: p.stat().st_mtime)
                snapshot_size = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())
        # Use the larger of the two: during download blobs >> snapshots (which
        # is empty); after completion snapshots == blobs (symlinks) or
        # snapshots > blobs (Windows copy mode where blobs is emptied).
        size = max(blobs_size, snapshot_size)
        if size == 0:
            continue

        # Snapshot path for display (may not exist yet during download)
        snapshot_path = ""
        if snapshots_dir.exists():
            snaps = list(snapshots_dir.iterdir())
            if snaps:
                snapshot_path = str(max(snaps, key=lambda p: p.stat().st_mtime))

        # GGUF enrichment: list .gguf files actually present in the latest snapshot
        gguf_files: Optional[List[str]] = None
        active_for_this: Optional[str] = None
        if is_gguf_repo(hf_id):
            gguf_files = []
            if snapshot_path:
                snap_p = Path(snapshot_path)
                gguf_files = sorted(
                    f.name for f in snap_p.iterdir() if f.is_file() and f.name.endswith(".gguf")
                )
            if hf_id == active_hf_id:
                active_for_this = active_gguf

        # Determine download_status: complete > downloading > paused
        # - complete: snapshots/<rev>/ has a non-incomplete file (download done,
        #   hf-xet linked the blob into snapshots)
        # - downloading: partial bytes exist AND manager is actively loading this id
        # - paused: partial bytes exist but no worker writing (e.g., backend
        #   restarted, or load was never re-triggered after configure)
        has_complete_files = False
        if snapshots_dir.exists():
            for snap in snapshots_dir.iterdir():
                if not snap.is_dir():
                    continue
                for f in snap.iterdir():
                    if f.is_file() and not f.name.endswith(".incomplete"):
                        has_complete_files = True
                        break
                if has_complete_files:
                    break

        if has_complete_files:
            download_status = "complete"
        elif active_state == "LOADING" and active_hf_id == hf_id:
            download_status = "downloading"
        else:
            download_status = "paused"

        out.append(CachedModelInfo(
            hf_id=hf_id,
            size_bytes=size,
            snapshot_path=snapshot_path,
            gguf_files=gguf_files,
            active_gguf_file=active_for_this,
            download_status=download_status,
            expected_size_bytes=None,  # populated below if HF API call succeeds
        ))
    return out


@router.delete("/cached")
def delete_cached(
    hf_id: str = Query(..., description="HF model id, e.g. Qwen/Qwen3-8B-Instruct")
) -> dict:
    """Delete a cached snapshot. Refuses if the model is currently loaded."""
    mgr = ModelManager.get()
    status = mgr.status()
    if status["hf_id"] == hf_id and status["state"] == "READY":
        raise HTTPException(
            status_code=409,
            detail=f"Model {hf_id} is currently loaded. Unload first before deleting.",
        )
    cache_subdir = HF_CACHE_DIR / f"models--{hf_id.replace('/', '--')}"
    if not cache_subdir.exists():
        raise HTTPException(status_code=404, detail=f"No cached snapshot found for {hf_id}")
    shutil.rmtree(cache_subdir)
    return {"ok": True, "deleted": str(cache_subdir)}
