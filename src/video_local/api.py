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
    prompt: str = (
        "Summer beach vacation style, a white cat wearing sunglasses sits "
        "on a surfboard. The cat gazes at the camera with a relaxed expression."
    )
    duration: float = 3.0
    sampling_steps: int = 30
    max_area: int = 480 * 480
    guide_scale: float = 5.0
    # Optional driver wav. If absent, defaults to Wan's bundled
    # examples/talk.wav so lipsync is observably exercised by the smoke
    # test. Pass an explicit path to test with CosyVoice output.
    audio_path: Optional[str] = None
    image_path: Optional[str] = None


@router.post("/test_synthesize")
async def test_synthesize(req: TestSynthRequest = TestSynthRequest()) -> dict:
    """Diagnostic — Wan2.2-S2V-14B with bundled image + audio (or
    caller-supplied paths). Output mp4 has lipsync audio muxed in."""
    project_root = Path(__file__).resolve().parents[2]
    wan_examples = project_root / "output" / "external" / "Wan2.2" / "examples"
    out_path = str(project_root / "output" / "test_video.mp4")

    image = req.image_path or str(wan_examples / "i2v_input.JPG")
    audio = req.audio_path or str(wan_examples / "talk.wav")

    # Map duration → frame_num (S2V sample_fps=16, infer_frames must be 4n).
    fps = 16
    frame_num = max(4, (int(round(float(req.duration) * fps)) // 4) * 4)

    mgr = VideoModelManager.get()
    try:
        path = await asyncio.to_thread(
            mgr.generate_native,
            image=image,
            audio=audio,
            prompt=req.prompt,
            output_path=out_path,
            frame_num=frame_num,
            sampling_steps=req.sampling_steps,
            max_area=req.max_area,
            guide_scale=req.guide_scale,
        )
        return {"ok": True, "path": path, "frame_num": frame_num}
    except Exception as e:
        logger.exception("test_synthesize failed")
        raise HTTPException(status_code=500, detail=str(e))
