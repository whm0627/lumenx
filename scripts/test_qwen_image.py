"""Standalone Qwen-Image smoke test — no backend, no frontend.

Phases (each timed and logged):
  1. instantiate LocalQwenImageModel
  2. _get_pipe() — auto-resumes missing HF cache files,
     constructs diffusers pipeline, applies torchao int8, .to('cuda')
  3. generate() — one 1024x1024 T2I

Run: .venv/Scripts/python.exe scripts/test_qwen_image.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Ensure HF_HOME points at our cache before any HF / diffusers import.
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / "output" / "models" / "LLM"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_qwen_image")

# Late import so the env var above is honoured.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.image_local import LocalQwenImageModel  # noqa: E402


def main() -> int:
    print("=" * 70, flush=True)
    print("Qwen-Image local smoke test", flush=True)
    print(f"HF_HOME = {os.environ.get('HF_HOME')}", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/3] Instantiating LocalQwenImageModel ...", flush=True)
    m = LocalQwenImageModel({})
    print(f"  hf_id={m.hf_id}, pipe loaded={m._pipe is not None}", flush=True)

    print("\n[2/3] Loading pipe (download remainder + diffusers init + int8 quant + to cuda) ...",
          flush=True)
    print("  This may take 3-10 minutes — file IO is the bottleneck.", flush=True)
    t0 = time.monotonic()
    try:
        m._get_pipe()
    except Exception as e:
        print(f"  LOAD FAILED after {time.monotonic()-t0:.1f}s: {type(e).__name__}: {e}",
              flush=True)
        logger.exception("load failed")
        return 1
    t_load = time.monotonic() - t0
    print(f"  load done in {t_load:.1f}s, pipe loaded={m._pipe is not None}",
          flush=True)

    print("\n[3/3] Generating one 1024x1024 T2I ...", flush=True)
    prompt = (
        "concept art of a young wizard with black hair and purple eyes, "
        "full body, standing pose, isolated on plain white background, "
        "high quality, masterpiece"
    )
    out_path = str(Path(__file__).resolve().parents[1] / "output" / "qwen_image_smoke.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        path, gen_dur = m.generate(prompt, out_path, size="1024*1024")
    except Exception as e:
        print(f"  GENERATE FAILED after {time.monotonic()-t0:.1f}s: {type(e).__name__}: {e}",
              flush=True)
        logger.exception("generate failed")
        return 2
    print(f"  generated in {gen_dur:.1f}s -> {path}", flush=True)
    print(f"  file size: {os.path.getsize(path):,} bytes", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"OK — load {t_load:.1f}s, generate {gen_dur:.1f}s, output={out_path}",
          flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
