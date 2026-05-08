"""LocalWanS2V — local video generation runtime, Wan2.2-S2V-14B with lipsync.

Wraps Wan2.2-S2V-14B (audio + image + text → video with **native lipsync**).
The DiT is loaded entirely from a GGUF file (Q4_K_S → ~7GB on GPU); the
upstream 31GB of bf16 DiT safetensors are SKIPPED — we only download
T5 + VAE + configs from Wan-AI/Wan2.2-S2V-14B (~11GB), and the GGUF
provides every DiT weight tensor (1260 tensors covering both quantized
Linears and small fp16/fp32 norms / modulations / embeddings).

Layout, in order:
  1. snapshot_download(Wan-AI/Wan2.2-S2V-14B) with allow_patterns that
     match T5 .pth, VAE .pth, all .json configs, but EXCLUDE
     `diffusion_pytorch_model-*.safetensors` (~31GB).
  2. snapshot_download(QuantStack/Wan2.2-S2V-14B-GGUF, *Q4_K_S.gguf).
  3. Hook `WanModel_S2V.from_pretrained` → build empty model from
     config.json only (no safetensors load), under
     `accelerate.init_empty_weights()`.
  4. Construct wan.WanS2V (loads T5 + VAE normally; DiT is empty meta).
  5. `_load_dit_from_gguf` — walk the empty DiT and:
        - Replace each nn.Linear that has a Q4_K/Q8_0 weight in GGUF
          with our `GGUFLinear` (in-house dequant kernel).
        - For every other named_parameter / named_buffer, copy the
          matching GGUF tensor's bytes in (F16/F32/BF16 → torch).
        - Log any tensors that didn't get filled (means GGUF lacked
          them — should be 0 for a well-formed Wan-S2V GGUF).
  6. `flash_attn_shim.patch_wan_flash_attention()` because the
     `flash_attn` package isn't available for torch 2.11+cu128 Windows.
     Wan-S2V calls flash_attention directly so we replace the symbol
     with a SDPA wrapper.

Generate path:
  Wan saves a silent mp4. We then ffmpeg-mux the driver wav into it
  (whatever audio drove the lipsync) so the output mp4 has a real audio
  track. If audio is the silent fallback (no dialogue), output stays
  silent.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

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


# Files we DO need from Wan-AI/Wan2.2-S2V-14B. Excludes
# `diffusion_pytorch_model-*.safetensors` (31GB) since GGUF supplies
# every DiT weight. Patterns match recursively into google/umt5-xxl/
# and wav2vec2-large-xlsr-53-english/.
_S2V_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    # Wav2Vec2 audio encoder used by Wan-S2V's casual_audio_encoder
    # for lipsync conditioning. Without this the load fails with
    # "Error no file named pytorch_model.bin, model.safetensors, ..."
    # at WanS2V construction. Using the .safetensors variant so we
    # don't also pull pytorch_model.bin / flax_model.msgpack (also
    # ~1.2GB each but redundant).
    "wav2vec2-large-xlsr-53-english/model.safetensors",
]

_GGUF_FILE_PATTERNS = {
    "Q8_0":   "*Q8_0.gguf",
    "Q4_K_S": "*Q4_K_S.gguf",
}
_SUPPORTED_QUANTS = set(_GGUF_FILE_PATTERNS.keys())


@contextmanager
def _stub_dit_load():
    """Hook `WanModel_S2V.from_pretrained` to skip safetensors load —
    we don't have them on disk (skipped during snapshot_download to save
    31GB) and don't need them (GGUF supplies all DiT weights).

    Builds the model from `config.json` under `init_empty_weights()` so
    construction touches no real memory. After this context exits and
    Wan finishes building the rest of WanS2V, our `_load_dit_from_gguf`
    materializes the model on CPU and fills weights from the GGUF.
    """
    _inject_wan_paths()
    from accelerate import init_empty_weights
    from wan.modules.s2v.model_s2v import WanModel_S2V

    original = WanModel_S2V.from_pretrained

    def stub_from_pretrained(cls, model_path, *args, **kwargs):
        # Diffusers' from_pretrained tries to read a sharded safetensors
        # index; without those files it errors. Side-step entirely by
        # using `from_config` (which only needs config.json).
        config = cls.load_config(str(model_path))
        with init_empty_weights():
            model = cls.from_config(config)
        # Mark for downstream patcher.
        setattr(model, "_lumenx_empty_meta", True)
        logger.info(
            "[LocalWanS2V] built empty WanModel_S2V from config "
            "(skipping bf16 safetensors load — GGUF will fill weights)"
        )
        return model

    WanModel_S2V.from_pretrained = classmethod(stub_from_pretrained)
    try:
        yield
    finally:
        WanModel_S2V.from_pretrained = original


def _decode_unquantized_gguf(raw_bytes: memoryview, shape, quant_name: str):
    """Decode a non-quantized GGUF tensor (F16 / F32 / BF16) to a torch
    tensor with the requested shape."""
    import numpy as np
    import torch

    if quant_name == "F32":
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape).copy()
        return torch.from_numpy(arr)
    if quant_name == "F16":
        arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape).copy()
        return torch.from_numpy(arr)
    if quant_name == "BF16":
        # numpy has no native bf16; reinterpret uint16 → torch.bfloat16.
        arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    raise ValueError(f"_decode_unquantized_gguf: unsupported {quant_name!r}")


class LocalWanS2V:
    """Wan2.2-S2V-14B local runtime, GGUF-only weight path."""

    S2V_HF_ID = "Wan-AI/Wan2.2-S2V-14B"
    GGUF_HF_ID = "QuantStack/Wan2.2-S2V-14B-GGUF"

    def __init__(self, quant: str = "Q4_K_S"):
        if quant not in _SUPPORTED_QUANTS:
            raise ValueError(
                f"Unsupported local video quant {quant!r}; expected "
                f"{sorted(_SUPPORTED_QUANTS)} (fp16 path requires the 31GB "
                f"DiT safetensors which we explicitly skip downloading)"
            )
        self.quant = quant
        self.hf_id: str = self.S2V_HF_ID
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

        logger.info(
            f"[LocalWanS2V] downloading {self.S2V_HF_ID} "
            f"(T5 + VAE + configs only — skipping 31GB DiT safetensors)"
        )
        wan_dir = snapshot_download(self.S2V_HF_ID, allow_patterns=_S2V_ALLOW_PATTERNS)

        pattern = _GGUF_FILE_PATTERNS[self.quant]
        logger.info(f"[LocalWanS2V] downloading {self.GGUF_HF_ID} {pattern}")
        gguf_dir = snapshot_download(self.GGUF_HF_ID, allow_patterns=[pattern])
        matches = sorted(Path(gguf_dir).glob(pattern))
        if not matches:
            raise RuntimeError(
                f"No GGUF file matching {pattern!r} found in {self.GGUF_HF_ID}"
            )
        gguf_path = str(matches[0])

        cfg = WAN_CONFIGS["s2v-14B"]
        logger.info(
            f"[LocalWanS2V] constructing wan.WanS2V "
            f"(quant={self.quant}, t5_cpu=True, init_on_cpu=True, "
            f"GGUF-only DiT)"
        )
        with _stub_dit_load():
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
                # Skip the post-load `model.to(bf16)` cast — our DiT is
                # currently meta-tensored, that cast errors. The GGUF
                # loader places weights at correct dtypes.
                convert_model_dtype=False,
            )

        self._load_dit_from_gguf(gguf_path)

        # Wan-S2V calls `flash_attention()` directly. flash_attn package
        # is not available for torch 2.11+cu128 Windows — patch the
        # symbol with our SDPA shim.
        from .flash_attn_shim import patch_wan_flash_attention
        patch_wan_flash_attention()

        if os.environ.get("USE_SAGE_ATTENTION") == "1":
            from .sage_attn import enable_sage_attention
            enable_sage_attention()

        logger.info(f"[LocalWanS2V] ready (S2V-14B {self.quant}, lipsync)")

    def _load_dit_from_gguf(self, gguf_path: str) -> None:
        """Materialize the meta-tensored noise_model on CPU and fill all
        weights from the GGUF.

        Two passes:
          1. nn.Linear modules whose weight is Q4_K / Q8_0 in GGUF →
             swap with our in-house GGUFLinear that holds the raw quant
             bytes and dequantizes per-forward.
          2. Remaining named_parameters + named_buffers → look up by
             name in GGUF, decode F32/F16/BF16, copy into the existing
             tensor slot (which is currently meta).
        Any param/buffer not in GGUF gets a real zeros allocation +
        a logged warning so it's debuggable if the model misbehaves.
        """
        import gc

        import torch
        import torch.nn as nn

        from .gguf.ops import GGUFLinear
        from .gguf.reader import parse_gguf

        logger.info(f"[LocalWanS2V] parsing GGUF {gguf_path}")
        gguf_tensors = parse_gguf(gguf_path)

        model = self._pipe.noise_model

        # ---------- pass 1: swap quantized Linears to GGUFLinear ----------
        replaced = 0
        for module_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            tensor_key = f"{module_name}.weight"
            if tensor_key not in gguf_tensors:
                continue
            ggt = gguf_tensors[tensor_key]
            # Quantized Linears handled by GGUFLinear in pass-1 — Q5_K
            # added for mixed-strategy GGUFs (QuantStack Q4_K_S uses
            # Q5_K for attention V + FFN.2). F16/F32/BF16 fall through
            # to pass-2 which loads them directly into nn.Linear.weight.
            if ggt.quant_type not in ("Q8_0", "Q4_K", "Q5_K"):
                continue
            # Bias for this Linear, if any, is a separate GGUF entry —
            # decode it and pass to the new module. Pass 2 won't touch
            # this since the parent module is gone.
            bias_clone: Optional[torch.Tensor] = None
            bias_key = f"{module_name}.bias"
            if module.bias is not None and bias_key in gguf_tensors:
                ggt_b = gguf_tensors[bias_key]
                if ggt_b.quant_type in ("F16", "F32", "BF16"):
                    bias_clone = _decode_unquantized_gguf(
                        ggt_b.raw_bytes, ggt_b.shape, ggt_b.quant_type,
                    ).to(torch.float16)
            new_linear = GGUFLinear(weight_tensor=ggt, bias=bias_clone)
            parent_name, _, attr_name = module_name.rpartition(".")
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, attr_name, new_linear)
            replaced += 1
        logger.info(
            f"[LocalWanS2V] pass-1: replaced {replaced} Linears with GGUFLinear"
        )

        # ---------- pass 2: fill remaining params + buffers ----------
        # Remaining params/buffers are still meta. Materialize on CPU
        # then overwrite from GGUF.
        filled = 0
        missing: list[str] = []
        for name, param in list(model.named_parameters()):
            if param.device.type != "meta":
                continue  # was filled by pass 1 (GGUFLinear has real CPU bias)
            ggt = gguf_tensors.get(name)
            if ggt is None:
                missing.append(name)
                # Allocate zeros so the model can run; sometimes this
                # is fine (zero-init bias), sometimes it's a load bug.
                with torch.no_grad():
                    new = torch.zeros(
                        param.shape, dtype=param.dtype, device="cpu",
                    )
                    self._set_param(model, name, new)
                continue
            if ggt.quant_type in ("Q8_0", "Q4_K", "Q5_K", "Q6_K"):
                missing.append(f"{name} (quantized but not a Linear weight)")
                with torch.no_grad():
                    new = torch.zeros(
                        param.shape, dtype=param.dtype, device="cpu",
                    )
                    self._set_param(model, name, new)
                continue
            tensor = _decode_unquantized_gguf(
                ggt.raw_bytes, ggt.shape, ggt.quant_type,
            )
            # Match destination dtype so `.to(bf16)` later doesn't blow up.
            dest_dtype = param.dtype if param.dtype != torch.float32 else tensor.dtype
            tensor = tensor.to(dest_dtype)
            self._set_param(model, name, tensor)
            filled += 1
        for name, buf in list(model.named_buffers()):
            if buf.device.type != "meta":
                continue
            ggt = gguf_tensors.get(name)
            if ggt is None:
                missing.append(f"buf:{name}")
                with torch.no_grad():
                    new = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                    self._set_buffer(model, name, new)
                continue
            tensor = _decode_unquantized_gguf(
                ggt.raw_bytes, ggt.shape, ggt.quant_type,
            )
            dest_dtype = buf.dtype if buf.dtype != torch.float32 else tensor.dtype
            tensor = tensor.to(dest_dtype)
            self._set_buffer(model, name, tensor)
            filled += 1

        logger.info(
            f"[LocalWanS2V] pass-2: filled {filled} non-Linear "
            f"params/buffers from GGUF; {len(missing)} missing"
        )
        if missing:
            sample = ", ".join(missing[:8])
            logger.warning(
                f"[LocalWanS2V] {len(missing)} tensors missing in GGUF "
                f"(zero-filled): {sample}{' ...' if len(missing) > 8 else ''}"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _set_param(model, dotted_name: str, tensor) -> None:
        """Replace `model.<dotted_name>` Parameter with a fresh
        Parameter wrapping `tensor`. Handles nested module paths."""
        import torch.nn as nn
        parent_name, _, attr = dotted_name.rpartition(".")
        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)
        setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))

    @staticmethod
    def _set_buffer(model, dotted_name: str, tensor) -> None:
        parent_name, _, attr = dotted_name.rpartition(".")
        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)
        # register_buffer overwrites in-place if the name exists
        parent.register_buffer(attr, tensor)

    def generate(self, image: str, audio: Optional[str], prompt: str, output_path: str,
                 max_area: int = 704 * 480, frame_num: int = 80,
                 sampling_steps: int = 30, guide_scale: float = 5.0,
                 shift: Optional[float] = None, sample_solver: str = "unipc",
                 negative_prompt: str = "",
                 seed: int = -1, init_first_frame: bool = False,
                 **kwargs) -> str:
        """Image+text+audio → video with lipsync, then ffmpeg-mux audio
        into the output mp4. Returns the muxed path."""
        if self._pipe is None:
            self.load()

        if shift is None:
            shift = 3.0 if max_area <= 704 * 480 else 5.0

        infer_frames = max(4, (frame_num // 4) * 4)
        from wan.configs import WAN_CONFIGS
        fps = WAN_CONFIGS["s2v-14B"].sample_fps
        duration_sec = infer_frames / fps

        external_audio = bool(audio and isinstance(audio, str) and os.path.exists(audio))
        if external_audio:
            driver_wav = audio
        else:
            if audio:
                logger.warning(
                    f"[LocalWanS2V] audio={audio!r} not found; using silent WAV"
                )
            driver_wav = self._make_silent_wav(duration_sec)

        offload_dit = os.environ.get("LOCAL_VIDEO_OFFLOAD", "0").strip() != "0"
        logger.info(
            f"[S2V-14B] audio={driver_wav} prompt={prompt!r} "
            f"infer_frames={infer_frames} steps={sampling_steps} "
            f"cfg={guide_scale} shift={shift} offload_model={offload_dit}"
        )
        video = self._pipe.generate(
            input_prompt=prompt,
            ref_image_path=image,
            audio_path=driver_wav,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=1,
            pose_video=None,
            max_area=max_area,
            infer_frames=infer_frames,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt or "",
            seed=seed,
            offload_model=offload_dit,
            init_first_frame=init_first_frame,
        )
        from wan.utils.utils import save_video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if external_audio:
            silent_path = output_path + ".video.mp4"
        else:
            silent_path = output_path

        save_video(
            tensor=video[None],
            save_file=silent_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        if external_audio:
            self._ffmpeg_mux(silent_path, driver_wav, output_path)
            try:
                os.remove(silent_path)
            except OSError:
                pass

        return output_path

    @staticmethod
    def _ffmpeg_mux(silent_mp4: str, audio_wav: str, out_mp4: str) -> None:
        """Mux audio wav into a silent mp4. Re-encodes audio to AAC,
        copies video stream as-is. Falls back to imageio_ffmpeg's bundled
        binary if `get_ffmpeg_path()` returns None (e.g. no system
        ffmpeg, no project-bundled one)."""
        ffmpeg = None
        try:
            from ..utils.system_check import get_ffmpeg_path
        except Exception:
            from src.utils.system_check import get_ffmpeg_path
        ffmpeg = get_ffmpeg_path()
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
        if not ffmpeg:
            logger.warning("[LocalWanS2V] no ffmpeg available; output mp4 stays silent")
            import shutil
            shutil.copyfile(silent_mp4, out_mp4)
            return

        cmd = [
            ffmpeg, "-y",
            "-i", silent_mp4,
            "-i", audio_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            out_mp4,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            logger.info(f"[LocalWanS2V] muxed audio → {out_mp4}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"[LocalWanS2V] ffmpeg mux failed (exit {e.returncode}); "
                f"stderr={e.stderr.decode(errors='replace')[:500]}"
            )
            import shutil
            shutil.copyfile(silent_mp4, out_mp4)

    @staticmethod
    def _make_silent_wav(duration_sec: float, sample_rate: int = 16000) -> str:
        import tempfile
        import wave

        n_frames = int(duration_sec * sample_rate)
        fd, path = tempfile.mkstemp(prefix="wan_s2v_silent_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * n_frames)
        return path

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            import gc

            import torch
            del self._pipe
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("[LocalWanS2V] unload best-effort cleanup failed")
