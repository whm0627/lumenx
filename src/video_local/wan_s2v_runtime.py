"""LocalWanS2V — local video generation runtime.

NOTE: Despite the historical class name (kept for API compatibility),
this currently wraps Wan2.2-TI2V-5B (image+text → video, NO lipsync),
not the originally-planned S2V-14B (audio-driven lipsync). Reasoning:

  * S2V-14B + GGUF Q4_K_S: model loaded but Python-per-layer dispatch
    overhead made inference 100% CPU-bound (~30+ min/clip, GPU 1-5%).
  * S2V-14B + torchao FP8: same dispatch overhead, same symptom.
  * S2V-14B + standard fp16 + offload: 23.8GB peak on 24GB ceiling,
    edge-of-OOM with no headroom.
  * TI2V-5B + standard fp16 + offload: 19GB peak, 100% GPU util,
    44s for 49-frame 704×480 clip on 4090. Works.

The GGUF infrastructure (gguf/reader.py, dequant kernels, GGUFLinear)
is left in place for a future S2V revisit but not used here. The
audio_path argument to generate() is accepted for API compatibility
but ignored (TI2V is text+image only). Lipsync is a follow-up.

Loading flow:
  1. snapshot_download Wan-AI/Wan2.2-TI2V-5B (T5 + VAE + DiT weights)
  2. Construct wan.WanTI2V with t5_cpu=False, init_on_cpu=True,
     convert_model_dtype=True

Inference flow:
  generate() proxies through to pipe.generate() with image+prompt.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _default_repo_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "output" / "external" / "Wan2.2"


def _inject_wan_paths() -> None:
    """Prepend the cloned Wan2.2 repo and its Matcha-TTS submodule to
    sys.path. Idempotent. Honors WAN_REPO_PATH env override."""
    repo = Path(os.environ.get("WAN_REPO_PATH", str(_default_repo_path())))
    if not repo.exists():
        return
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (repo, matcha):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# Pattern for GGUF tier filename. Matches QuantStack's naming convention.
_GGUF_FILE_PATTERNS = {
    "Q8_0":   "*Q8_0.gguf",
    "Q4_K_S": "*Q4_K_S.gguf",
}


class LocalWanS2V:
    def __init__(self, quant: str = "fp16"):
        # `quant` is accepted for API compat with the original GGUF-based
        # design but ignored — TI2V-5B fits in fp16 on a 24GB card with
        # offload_model=True. Future S2V revisit may reintroduce GGUF.
        self.quant = quant
        self.hf_id = "Wan-AI/Wan2.2-TI2V-5B"
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        _inject_wan_paths()
        try:
            import wan
            from wan.configs import WAN_CONFIGS
        except ImportError as e:
            repo = os.environ.get("WAN_REPO_PATH", str(_default_repo_path()))
            raise RuntimeError(
                f"Wan2.2 repo not importable from {repo!r}. Clone "
                f"github.com/Wan-Video/Wan2.2 to that path. Original error: {e}"
            ) from e

        from huggingface_hub import snapshot_download

        # 1. Download Wan2.2-TI2V-5B (T5 + VAE + DiT, ~32GB cache on disk
        # due to Windows symlink-fallback double-storing).
        logger.info(f"[LocalWanS2V] downloading {self.hf_id} (~32 GB cache if not present)")
        wan_dir = snapshot_download(self.hf_id)

        # 2. Construct WanTI2V. t5_cpu=False is critical: TI2V's encode
        # path runs CFG (positive + negative prompt → 2× T5 calls); on
        # CPU a single umt5-xxl pass is ~30s, two encodes hung 30+ min
        # observed. With t5_cpu=False, T5 briefly takes ~11GB GPU during
        # encode, then offload_model=True swaps it back to CPU and the
        # DiT (~10GB) takes its place for the diffusion loop. Peak ~19GB
        # on a 24GB card with headroom for activations.
        cfg = WAN_CONFIGS["ti2v-5B"]
        logger.info("[LocalWanS2V] constructing wan.WanTI2V (t5_cpu=False, init_on_cpu=True)")
        self._pipe = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=wan_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=True,
        )

        # 3. Sage attention if requested
        if os.environ.get("USE_SAGE_ATTENTION") == "1":
            from .sage_attn import enable_sage_attention
            enable_sage_attention()

        logger.info(f"[LocalWanS2V] ready (TI2V-5B fp16)")

    def _patch_with_gguf(self, gguf_path: str) -> None:
        """Walk pipe.noise_model and replace each nn.Linear that has a
        matching GGUF tensor with GGUFLinear carrying the quantized
        bytes. Linears that weren't quantized in the GGUF (e.g. small
        embedding/norm layers) keep their fp16 weights from the bf16
        safetensors load."""
        import torch.nn as nn

        from .gguf.ops import GGUFLinear
        from .gguf.reader import parse_gguf

        logger.info(f"[LocalWanS2V] parsing GGUF {gguf_path}")
        gguf_tensors = parse_gguf(gguf_path)

        replaced = 0
        kept = 0
        for module_name, module in list(self._pipe.noise_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # GGUF tensor name convention: f"{module_name}.weight"
            tensor_key = f"{module_name}.weight"
            if tensor_key not in gguf_tensors:
                kept += 1
                continue
            ggt = gguf_tensors[tensor_key]
            # GGML block types we wrap. Note Q4_K covers both Q4_K_S and
            # Q4_K_M strategies — the file-name suffix is just a quant
            # rounding policy, the block layout is the same.
            if ggt.quant_type not in ("Q8_0", "Q4_K"):
                # F16 / F32 / BF16 — leave as plain nn.Linear with weights
                # already loaded from the safetensors snapshot.
                kept += 1
                continue
            # Detach the bias to a fresh fp16 tensor so the parent module's
            # original Linear's nn.Parameter (with the bf16 weight) becomes
            # unreferenced once we setattr-replace below — letting Python
            # GC + torch.cuda.empty_cache() actually free those bf16 weights.
            bias_clone = module.bias.detach().clone() if module.bias is not None else None
            new_linear = GGUFLinear(weight_tensor=ggt, bias=bias_clone)
            # Re-attach via setattr on parent
            parent_name, _, attr_name = module_name.rpartition(".")
            parent = self._pipe.noise_model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, attr_name, new_linear)
            replaced += 1
        logger.info(f"[LocalWanS2V] replaced {replaced} Linears with GGUFLinear; "
                    f"kept {kept} (no GGUF counterpart or F16)")
        # Drop any GC'd bf16 weights and force CUDA to release that block back to
        # the allocator. Without this, the original 14B bf16 weights stay
        # cached on GPU even after the swap, doubling resident VRAM (~24GB
        # observed instead of the expected ~13GB for Q4_K_S).
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, image: str, audio: Optional[str], prompt: str, output_path: str,
                 max_area: int = 704 * 480, frame_num: int = 49,
                 sampling_steps: int = 30, guide_scale: float = 5.0,
                 seed: int = -1, **kwargs) -> str:
        """Image+text → video.

        The `audio` arg is part of the API contract for future lipsync
        post-processing (e.g. LatentSync / SadTalker layered on top of
        the silent TI2V output, OR a future S2V revisit). TI2V-5B itself
        is text+image only — we don't pass audio into Wan's pipe.
        """
        if self._pipe is None:
            raise RuntimeError("LocalWanS2V.generate called before load()")

        from PIL import Image
        img = Image.open(image).convert("RGB")
        if audio is not None:
            logger.info(f"[LocalWanS2V] audio={audio!r} not used by TI2V-5B; "
                        "preserved for future lipsync post-processing")

        logger.info(f"[LocalWanS2V] generate prompt={prompt!r} frames={frame_num}")
        video = self._pipe.generate(
            input_prompt=prompt,
            img=img,
            max_area=max_area,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=True,
        )
        # Wan returns 4D tensor (C, F, H, W); save_video expects 5D batched
        # (B, C, F, H, W). Add batch dim with [None] + nrow=1 — same as the
        # official generate.py.
        from wan.utils.utils import save_video
        from wan.configs import WAN_CONFIGS
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=WAN_CONFIGS["ti2v-5B"].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return output_path

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            import torch
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("[LocalWanS2V] unload best-effort cleanup failed")
