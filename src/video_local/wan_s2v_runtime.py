"""LocalWanS2V — wraps Wan official wan.WanS2V pipeline, optionally
swapping the DiT's nn.Linear modules for GGUFLinear.

Loading flow:
  1. snapshot_download Wan-AI/Wan2.2-S2V-14B (T5/VAE/wav2vec/yaml/etc)
  2. If quant != fp16: snapshot_download QuantStack/Wan2.2-S2V-14B-GGUF
     restricted to the matching tier file
  3. Construct wan.WanS2V (which loads bf16 DiT weights via standard path)
  4. Walk pipe.noise_model.named_modules(); for each nn.Linear with a
     matching GGUF tensor, replace with GGUFLinear
  5. Move quantized DiT to CUDA
  6. If env USE_SAGE_ATTENTION=1, enable sage_attn shim

Inference flow:
  - generate() proxies through to pipe.generate(...) with the kwargs
    we've validated work end-to-end (frame_num=4n+1, max_area, etc.)
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
    def __init__(self, quant: str = "Q4_K_S"):
        if quant not in ("fp16", "Q8_0", "Q4_K_S"):
            raise ValueError(f"unsupported quant {quant!r}; want fp16|Q8_0|Q4_K_S")
        self.quant = quant
        self.hf_id = "Wan-AI/Wan2.2-S2V-14B"
        self.gguf_hf_id = "QuantStack/Wan2.2-S2V-14B-GGUF"
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

        # 1. Download original Wan2.2-S2V-14B (T5, VAE, wav2vec, yaml — DiT
        # safetensors will get downloaded too even though we may replace them).
        logger.info(f"[LocalWanS2V] downloading {self.hf_id} (~46 GB if not cached)")
        wan_dir = snapshot_download(self.hf_id)

        # 2. Download GGUF tier file if quantizing
        gguf_path: Optional[str] = None
        if self.quant != "fp16":
            pattern = _GGUF_FILE_PATTERNS[self.quant]
            logger.info(f"[LocalWanS2V] downloading {self.gguf_hf_id} ({pattern})")
            gguf_dir = snapshot_download(self.gguf_hf_id, allow_patterns=[pattern])
            matches = list(Path(gguf_dir).glob(pattern))
            if not matches:
                raise RuntimeError(f"no GGUF file matching {pattern} in {gguf_dir}")
            gguf_path = str(matches[0])

        # 3. Construct WanS2V (loads everything including bf16 DiT)
        cfg = WAN_CONFIGS["s2v-14B"]
        logger.info("[LocalWanS2V] constructing wan.WanS2V — t5_cpu=True, init_on_cpu=True")
        self._pipe = wan.WanS2V(
            config=cfg,
            checkpoint_dir=wan_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )

        # 4. If quantized, replace DiT linears with GGUFLinear
        if self.quant != "fp16" and gguf_path is not None:
            self._patch_with_gguf(gguf_path)

        # 5. Move (possibly quantized) DiT to GPU
        import torch
        self._pipe.noise_model = self._pipe.noise_model.to("cuda:0")
        torch.cuda.synchronize()

        # 6. Sage attention if requested
        if os.environ.get("USE_SAGE_ATTENTION") == "1":
            from .sage_attn import enable_sage_attention
            enable_sage_attention()

        logger.info(f"[LocalWanS2V] ready (quant={self.quant})")

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
            if ggt.quant_type == "F16":
                # GGUFLinear doesn't handle F16; leave the original linear
                kept += 1
                continue
            new_linear = GGUFLinear(
                weight_tensor=ggt,
                bias=module.bias.detach() if module.bias is not None else None,
            )
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

    def generate(self, image: str, audio: str, prompt: str, output_path: str,
                 max_area: int = 480 * 480, infer_frames: int = 21,
                 sampling_steps: int = 15, guide_scale: float = 1.0,
                 seed: int = -1, **kwargs) -> str:
        if self._pipe is None:
            raise RuntimeError("LocalWanS2V.generate called before load()")

        logger.info(f"[LocalWanS2V] generate prompt={prompt!r} frames={infer_frames}")
        video = self._pipe.generate(
            input_prompt=prompt,
            ref_image_path=image,
            audio_path=audio,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            max_area=max_area,
            infer_frames=infer_frames,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
        )
        # Wan returns a video tensor — save via repo's helper
        from wan.utils.utils import save_video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(video, output_path, fps=16)  # Wan default sample_fps
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
