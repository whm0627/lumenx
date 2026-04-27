"""Local Qwen-Image image generation, drop-in for ImageGenModel.

Loads Qwen/Qwen-Image via diffusers on first call, applies torchao
int8_weight_only quantization to fit a 24GB card with headroom for an
LLM swap, and exposes the same generate() signature as WanxImageModel
so AssetGenerator can swap providers transparently.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..utils import get_logger
from ..utils.gpu_lock import GPULock
from .image import ImageGenModel

logger = get_logger(__name__)

DEFAULT_HF_ID = "Qwen/Qwen-Image"
# Reference-conditioned generation (variants / view sheets / poses keeping
# the same character) needs Qwen-Image-Edit-2509 — the base Qwen-Image's
# Img2Img only does denoise-on-top-of-input, which produces "lightly modified
# original" rather than a new composition. Qwen-Image-Edit is the purpose-
# built model with multi-image (1–3 ref) support and IP creation / view
# rotation in its training mix.
DEFAULT_EDIT_HF_ID = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_SIZE = "1024*1024"
GPU_LOCK_NAME = "image"


class LocalQwenImageModel(ImageGenModel):
    """In-process Qwen-Image runtime via diffusers + torchao int8.

    Registers with GPULock so the LLM (or any other GPU runtime) is
    evicted when this model loads. unload() releases VRAM and the lock.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        params = config.get("params", {}) if isinstance(config, dict) else {}
        self.hf_id: str = params.get("hf_id", DEFAULT_HF_ID)
        self.edit_hf_id: str = params.get("edit_hf_id", DEFAULT_EDIT_HF_ID)
        # T2I pipeline (QwenImagePipeline) — loaded on first refs-less generate.
        self._pipe = None
        # Edit pipeline (QwenImageEditPlusPipeline) over Qwen-Image-Edit-2509 —
        # loaded lazily on first reference-conditioned generate. Independent
        # from _pipe (different model weights, separate from_pretrained call).
        self._pipe_edit = None
        # The hf_id of whichever pipe is currently in use; surfaced by the
        # manager's status() so the footer shows the actual model the user
        # is waiting on, not the unrelated T2I default.
        self.active_hf_id: str = self.hf_id
        GPULock.get().register(GPU_LOCK_NAME, self.unload)

    def _apply_vram_strategy(self, pipe, strategy: str) -> None:
        """Quantize (where compatible) + install offload hooks + tweak VAE.
        Shared between the T2I and Edit pipes so both follow the same
        auto-tuned VRAM budget."""
        if strategy != "sequential_offload":
            try:
                from torchao.quantization import int8_weight_only, quantize_

                quantize_(pipe.transformer, int8_weight_only())
                logger.info("Applied torchao int8_weight_only to transformer")
            except ImportError:
                logger.warning(
                    "torchao not installed; running bf16 (will need ~40GB VRAM)"
                )
        else:
            logger.info("Skipping torchao int8 — incompatible with sequential_cpu_offload")

        if strategy == "full_gpu":
            pipe.to("cuda")
        elif strategy == "model_offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()

        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            elif hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

    def _construct_pipe(self):
        """Build the T2I pipeline (Qwen-Image base via QwenImagePipeline)."""
        import torch
        from diffusers import QwenImagePipeline

        strategy = self._pick_offload_strategy()
        logger.info(f"Loading {self.hf_id} via diffusers (strategy={strategy})")
        pipe = QwenImagePipeline.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16
        )
        self._apply_vram_strategy(pipe, strategy)
        return pipe

    def _construct_edit_pipe(self):
        """Build the Edit pipeline (Qwen-Image-Edit-2509 via
        QwenImageEditPlusPipeline). Used for reference-conditioned
        generation — character variants, three-view sheets, pose changes —
        where Img2Img would just denoise on top of the input."""
        import torch
        from diffusers import QwenImageEditPlusPipeline

        strategy = self._pick_offload_strategy()
        logger.info(f"Loading {self.edit_hf_id} via diffusers (strategy={strategy})")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.edit_hf_id, torch_dtype=torch.bfloat16
        )
        self._apply_vram_strategy(pipe, strategy)
        return pipe

    def _get_edit_pipe(self):
        """Lazy-load the Edit pipeline. Different model weights from the
        T2I pipe, so first call triggers a separate ~17GB download."""
        self.active_hf_id = self.edit_hf_id
        if self._pipe_edit is not None:
            return self._pipe_edit
        GPULock.get().acquire(GPU_LOCK_NAME)
        self._pipe_edit = self._construct_edit_pipe()
        return self._pipe_edit

    @staticmethod
    def _pick_offload_strategy() -> str:
        """Choose offload mode based on detected VRAM.

        Breakpoints calibrated against observed Qwen-Image peak usage:
        - >= 48GB (A6000, A100, H100): full_gpu — no offload, max throughput.
        - 32-48GB (RTX 6000 Ada, etc.): model_offload — transformer GPU-
          resident, encoders/VAE swap. ~22-23GB peak on I2I fits with margin.
        - < 32GB (4090, 3090, 24GB and below): sequential_offload — every
          submodule swaps on demand. Empirically the only safe choice on a
          24GB card; model_offload was observed to peak at ~23.7GB during
          I2I, leaving no headroom for PyTorch's allocator cache and risking
          OOM on the next generation.
        Override with LOCAL_IMG_OFFLOAD env var ("full_gpu" / "model_offload"
        / "sequential_offload") if you have a different memory budget.
        """
        import os

        override = os.getenv("LOCAL_IMG_OFFLOAD", "").strip().lower()
        if override in {"full_gpu", "model_offload", "sequential_offload"}:
            logger.info(f"Offload strategy from env: {override}")
            return override
        try:
            import torch

            if not torch.cuda.is_available():
                return "sequential_offload"
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            logger.info(f"Detected GPU VRAM: {vram_gb:.1f} GB")
            if vram_gb >= 48:
                return "full_gpu"
            if vram_gb >= 32:
                return "model_offload"
            return "sequential_offload"
        except Exception:
            logger.exception("VRAM detection failed; defaulting to sequential offload")
            return "sequential_offload"

    def _load_pipe(self):
        # Claim the GPU before allocating any VRAM — evicts any prior holder
        # (typically the LLM) so we don't OOM on a 24GB card.
        GPULock.get().acquire(GPU_LOCK_NAME)
        return self._construct_pipe()

    def _get_pipe(self):
        self.active_hf_id = self.hf_id
        if self._pipe is None:
            self._pipe = self._load_pipe()
        return self._pipe

    def unload(self) -> None:
        """Drop the pipelines, free CUDA cache, release the GPU lock.
        Idempotent: safe to call when nothing is loaded."""
        if self._pipe is not None:
            logger.info(f"Unloading {self.hf_id}")
            self._pipe = None
            self._pipe_edit = None  # release Edit pipe alongside T2I pipe
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        GPULock.get().release(GPU_LOCK_NAME)

    @staticmethod
    def _parse_size(size: str) -> Tuple[int, int]:
        # Wanx uses "WIDTH*HEIGHT", e.g. "576*1024"; preserve that contract.
        w, h = size.split("*")
        return int(w), int(h)

    @staticmethod
    def _collect_refs(
        ref_image_path: Optional[str],
        ref_image_paths: Optional[List[str]],
    ) -> List[str]:
        out: List[str] = []
        if ref_image_path:
            out.append(ref_image_path)
        if ref_image_paths:
            out.extend(ref_image_paths)
        # Preserve order while deduplicating
        seen = set()
        unique = []
        for p in out:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def generate(
        self,
        prompt: str,
        output_path: str,
        ref_image_path: Optional[str] = None,
        ref_image_paths: Optional[List[str]] = None,
        size: str = DEFAULT_SIZE,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        progress_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[str, float]:
        """Generate an image. If progress_callback is provided, it will be
        invoked once per inference step with `(step_index, total_steps)` so
        an outer manager can surface step-level progress to the UI."""
        t0 = time.monotonic()
        width, height = self._parse_size(size)
        refs = self._collect_refs(ref_image_path, ref_image_paths)

        # No refs → T2I (Qwen-Image base). With refs → Qwen-Image-Edit-2509,
        # which is purpose-built for reference-conditioned generation
        # (variants, view sheets, pose changes) where Img2Img would just
        # denoise on top of the input.
        if refs:
            pipe = self._get_edit_pipe()
        else:
            pipe = self._get_pipe()

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        if refs:
            loaded = [Image.open(p).convert("RGB") for p in refs]
            call_kwargs["image"] = loaded[0] if len(loaded) == 1 else loaded

        # Progress hook: wrap scheduler.step instead of using diffusers'
        # callback_on_step_end. The callback hook is NOT invoked by
        # QwenImageImg2ImgPipeline in diffusers 0.37 (verified empirically
        # — full_body T2I works, three_view I2I sits at progress=0 even
        # though the GPU is at 100% util). scheduler.step is called once
        # per denoising step regardless of pipeline class, so wrapping it
        # gives a uniform progress signal for both T2I and I2I paths.
        original_step = None
        scheduler = getattr(pipe, "scheduler", None)
        if progress_callback is not None and scheduler is not None:
            original_step = scheduler.step
            counter = {"i": 0}

            def _wrapped_step(*args, **kwargs):
                counter["i"] += 1
                # Use the scheduler's actual timestep count (which already
                # accounts for I2I strength scaling) so progress reaches
                # 100% rather than topping out at strength*100%.
                total = len(getattr(scheduler, "timesteps", [])) or num_inference_steps
                try:
                    progress_callback(counter["i"], total)
                except Exception:
                    logger.exception("progress_callback raised")
                return original_step(*args, **kwargs)

            scheduler.step = _wrapped_step

        try:
            result = pipe(**call_kwargs)
        finally:
            # Restore the scheduler so subsequent generates (or shared
            # components reused by another pipeline class) don't keep our
            # wrapper around.
            if original_step is not None and scheduler is not None:
                scheduler.step = original_step
        result.images[0].save(output_path)
        return output_path, time.monotonic() - t0
