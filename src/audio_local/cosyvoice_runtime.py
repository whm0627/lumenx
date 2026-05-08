"""LocalCosyVoiceTTS: project wrapper for CosyVoice-300M-SFT.

The local TTS path intentionally uses the SFT model, not zero-shot
voice cloning. It needs no reference audio: callers select one of the model's
built-in speaker IDs and we call inference_sft(text, speaker).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_repo_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "output" / "external" / "CosyVoice"


def _inject_cosyvoice_paths() -> None:
    repo = Path(os.environ.get("COSYVOICE_REPO_PATH", str(_default_repo_path())))
    if not repo.exists():
        return
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (repo, matcha):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


COSYVOICE_SFT_PRESETS = [
    {"id": "中文女", "name": "中文女声", "gender": "Female", "lang": "zh"},
    {"id": "中文男", "name": "中文男声", "gender": "Male", "lang": "zh"},
    {"id": "日语男", "name": "日语男声", "gender": "Male", "lang": "ja"},
    {"id": "粤语女", "name": "粤语女声", "gender": "Female", "lang": "zh-yue"},
    {"id": "英文女", "name": "English Female", "gender": "Female", "lang": "en"},
    {"id": "英文男", "name": "English Male", "gender": "Male", "lang": "en"},
    {"id": "韩语女", "name": "한국어 여성", "gender": "Female", "lang": "ko"},
]

DEFAULT_COSYVOICE_HF_ID = os.getenv(
    "LOCAL_TTS_HF_ID", "FunAudioLLM/CosyVoice-300M-SFT"
)


class LocalCosyVoiceTTS:
    """Lazy-loading wrapper around CosyVoice-300M-SFT inference."""

    def __init__(self, hf_id: str = DEFAULT_COSYVOICE_HF_ID, device: Optional[str] = None):
        self.hf_id = hf_id
        self.device = device or self._pick_device()
        self._cosyvoice: Any = None
        self._sample_rate: int = 22050
        self.active_label: str = ""

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self) -> None:
        if self._cosyvoice is not None:
            return

        _inject_cosyvoice_paths()
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
        except ImportError as e:
            repo = os.environ.get("COSYVOICE_REPO_PATH", str(_default_repo_path()))
            raise RuntimeError(
                f"CosyVoice repo not importable from {repo!r}. Clone "
                "github.com/FunAudioLLM/CosyVoice with submodules, or set "
                f"COSYVOICE_REPO_PATH. Original error: {e}"
            ) from e

        from huggingface_hub import snapshot_download

        logger.info("[LocalCosyVoiceTTS] downloading %s (if not cached)", self.hf_id)
        local_dir = snapshot_download(repo_id=self.hf_id)
        logger.info(
            "[LocalCosyVoiceTTS] weights at %s, constructing CosyVoice SFT",
            local_dir,
        )
        self._cosyvoice = CosyVoice(
            local_dir, load_jit=False, load_trt=False, fp16=False
        )
        self._sample_rate = getattr(self._cosyvoice, "sample_rate", 22050)
        logger.info(
            "[LocalCosyVoiceTTS] ready (device=%s, sr=%s, speakers=%s)",
            self.device,
            self._sample_rate,
            self.list_available_spks(),
        )

    def list_available_spks(self) -> List[str]:
        if self._cosyvoice is None:
            return [v["id"] for v in COSYVOICE_SFT_PRESETS]
        try:
            return list(self._cosyvoice.list_available_spks())
        except Exception:
            logger.exception("list_available_spks failed; falling back to static list")
            return [v["id"] for v in COSYVOICE_SFT_PRESETS]

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
        volume: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[str, float, str]:
        """Synthesize text to a PCM WAV and return (path, delay_ms, request_id)."""
        import time

        if self._cosyvoice is None:
            self.load()

        voice = voice or "中文女"
        available = set(self.list_available_spks())
        if voice not in available:
            raise ValueError(
                f"Unsupported local CosyVoice SFT voice {voice!r}; "
                f"expected one of {sorted(available)}"
            )

        self.active_label = f"{voice}: {text[:24]}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info("[LocalCosyVoiceTTS] synthesize voice=%s chars=%s", voice, len(text))

        start = time.time()
        first_package_delay_ms = 0.0
        chunks = []
        gen = self._cosyvoice.inference_sft(
            text,
            voice,
            stream=False,
            speed=speech_rate,
        )
        idx = 0
        for item in gen:
            idx += 1
            if first_package_delay_ms == 0.0:
                first_package_delay_ms = (time.time() - start) * 1000.0
            chunks.append(self._extract_audio_tensor(item))
            if progress_callback:
                try:
                    progress_callback(idx, idx + 1)
                except Exception:
                    logger.exception("progress_callback raised")

        if not chunks:
            raise RuntimeError("CosyVoice SFT produced no audio chunks for this input")

        self._write_wav(chunks, output_path)
        if progress_callback:
            try:
                progress_callback(idx, idx)
            except Exception:
                logger.exception("progress_callback raised on final tick")

        request_id = f"local-cosyvoice-300m-sft-{int(start * 1000)}"
        logger.info(
            "[LocalCosyVoiceTTS] done: chunks=%s first_chunk=%.0fms total=%.2fs",
            len(chunks),
            first_package_delay_ms,
            time.time() - start,
        )
        return output_path, first_package_delay_ms, request_id

    @staticmethod
    def _extract_audio_tensor(item: Any) -> Any:
        if isinstance(item, dict):
            for key in ("tts_speech", "speech", "audio"):
                if key in item:
                    return item[key]
        if isinstance(item, tuple) and len(item) == 2:
            return item[1]
        return item

    def _write_wav(self, chunks: List[Any], output_path: str) -> None:
        import torch
        import torchaudio

        waveform = torch.cat([self._to_2d(c) for c in chunks], dim=1).cpu()
        waveform_int16 = (waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
        torchaudio.save(
            output_path,
            waveform_int16,
            self._sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )

    @staticmethod
    def _to_2d(tensor: Any) -> Any:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def unload(self) -> None:
        if self._cosyvoice is None:
            return
        try:
            import torch

            del self._cosyvoice
            self._cosyvoice = None
            self.active_label = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("LocalCosyVoiceTTS.unload: best-effort cleanup failed")
