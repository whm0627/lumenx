"""FastAPI router exposing /audio/local/* endpoints for the global
status footer and explicit user-driven load/cancel.

Mirrors src/img_local/api.py: status + load + cancel + a small
diagnostic synthesize. No /configure or /cached yet — CosyVoice-300M-SFT
is the only local TTS model we support and the modal hardcodes it.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .manager import AudioModelManager, AudioState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio/local", tags=["local-audio"])


@router.get("/status")
def get_status() -> dict:
    return AudioModelManager.get().status()


@router.get("/voices")
def list_voices() -> dict:
    """Voice presets the local model accepts — what the picker UI shows
    when TTS_PROVIDER=local."""
    return {"voices": AudioModelManager.get().list_voices()}


@router.post("/load")
async def load() -> dict:
    """Force a load of the local TTS runtime. Triggers HF download +
    CosyVoice SFT init on first call. Idempotent if already READY."""
    try:
        return await AudioModelManager.get().load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel() -> dict:
    """Best-effort: drop the model + clear ERROR state. Currently does
    NOT interrupt an in-flight HF download (no progress hooks yet)."""
    mgr = AudioModelManager.get()
    s = mgr.status()
    if s["state"] == AudioState.UNLOADED.value:
        return {"ok": True, "cancelled": False}
    await mgr.unload()
    return {"ok": True, "cancelled": True}


@router.post("/test_synthesize")
async def test_synthesize(
    text: str = "你好，这是一段本地合成的中文男声测试。",
    voice: str = "中文男",
) -> dict:
    """Diagnostic: small synth so /status can be polled to verify the
    progress + state machine are firing. Saves to output/test_tts.wav."""
    out_path = str(
        Path(__file__).resolve().parents[2] / "output" / "test_tts.wav"
    )
    mgr = AudioModelManager.get()
    mgr.set_generation_label(f"test {voice}")
    try:
        path, delay_ms, rid = await asyncio.to_thread(
            mgr.synthesize, text, out_path, voice
        )
        return {
            "ok": True,
            "path": path,
            "first_package_delay_ms": round(delay_ms, 1),
            "request_id": rid,
        }
    except Exception as e:
        logger.exception("test_synthesize failed")
        raise HTTPException(status_code=500, detail=str(e))
