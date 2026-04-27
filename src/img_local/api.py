"""FastAPI router exposing /img/local/* endpoints for the global status
footer and explicit user-driven load/cancel.

Mirrors src/llm_local/api.py at a smaller scope: status + load + cancel.
No /configure or /cached yet — Qwen-Image is the only local image
model we support, and the modal hardcodes it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .manager import ImageModelManager, ImageState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/img/local", tags=["local-image"])


@router.get("/status")
def get_status() -> dict:
    return ImageModelManager.get().status()


@router.post("/load")
async def load() -> dict:
    """Force a load of the local image runtime. Triggers HF download +
    diffusers init + torchao quant on first call. Idempotent if already
    READY. Returns the resulting status (or 500 with the error
    captured in /status)."""
    try:
        return await ImageModelManager.get().load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel() -> dict:
    """Best-effort: drop pipe + clear ERROR state. Currently does NOT
    interrupt an in-flight HF download (no progress hooks yet)."""
    mgr = ImageModelManager.get()
    s = mgr.status()
    if s["state"] == ImageState.UNLOADED.value:
        return {"ok": True, "cancelled": False}
    await mgr.unload()
    return {"ok": True, "cancelled": True}


@router.post("/test_generate")
async def test_generate(
    prompt: str = "a single red apple on a white background",
    steps: int = 20,
    ref_path: str = "",
) -> dict:
    """Diagnostic: small generate so /status can be polled to verify the
    progress hook + Edit-pipe routing are firing. Saves to output/ as
    test_t2i.png (T2I) or test_i2i.png (I2I). Pass ref_path=output/foo.png
    to exercise the I2I path through Qwen-Image-Edit."""
    import asyncio
    from pathlib import Path

    is_i2i = bool(ref_path)
    out_name = "test_i2i.png" if is_i2i else "test_t2i.png"
    out_path = str(Path(__file__).resolve().parents[2] / "output" / out_name)
    mgr = ImageModelManager.get()
    mgr.set_generation_label("test_i2i" if is_i2i else "test_t2i")
    try:
        kwargs = {"num_inference_steps": steps, "size": "768*768"}
        if is_i2i:
            kwargs["ref_image_path"] = ref_path
        result = await asyncio.to_thread(mgr.generate, prompt, out_path, **kwargs)
        path, dur = result
        return {"ok": True, "mode": "i2i" if is_i2i else "t2i",
                "path": path, "duration_sec": round(dur, 2), "steps": steps}
    except Exception as e:
        logger.exception("test_generate failed")
        raise HTTPException(status_code=500, detail=str(e))
