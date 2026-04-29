"""FastAPI router for /video/local/* endpoints. Mirrors src/audio_local/api.py."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .manager import VideoModelManager, VideoState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video/local", tags=["local-video"])


class LoadRequest(BaseModel):
    quant: Optional[str] = None


@router.get("/status")
def get_status() -> dict:
    return VideoModelManager.get().status()


@router.post("/load")
async def load(req: LoadRequest = LoadRequest()) -> dict:
    try:
        return await VideoModelManager.get().load(quant=req.quant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel() -> dict:
    mgr = VideoModelManager.get()
    s = mgr.status()
    if s["state"] == VideoState.UNLOADED.value:
        return {"ok": True, "cancelled": False}
    await mgr.unload()
    return {"ok": True, "cancelled": True}


class TestSynthRequest(BaseModel):
    # Default prompt matches the bundled examples/i2v_input.JPG (sunglasses
    # cat on a surfboard) so the smoke test isn't fighting a content mismatch.
    prompt: str = (
        "Summer beach vacation style, a white cat wearing sunglasses sits "
        "on a surfboard. The cat gazes at the camera with a relaxed expression. "
        "Crystal-clear water, distant green hills, blue sky."
    )


@router.post("/test_synthesize")
async def test_synthesize(req: TestSynthRequest = TestSynthRequest()) -> dict:
    """Diagnostic endpoint — synthesize a short 704×480 clip via TI2V-5B
    using the bundled examples/i2v_input.JPG + examples/talk.wav."""
    project_root = Path(__file__).resolve().parents[2]
    wan_examples = project_root / "output" / "external" / "Wan2.2" / "examples"
    out_path = str(project_root / "output" / "test_video.mp4")

    mgr = VideoModelManager.get()
    try:
        path = await asyncio.to_thread(
            mgr.generate,
            image=str(wan_examples / "i2v_input.JPG"),
            audio=str(wan_examples / "talk.wav"),
            prompt=req.prompt,
            output_path=out_path,
        )
        return {"ok": True, "path": path}
    except Exception as e:
        logger.exception("test_synthesize failed")
        raise HTTPException(status_code=500, detail=str(e))
