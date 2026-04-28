"""LocalCosyVoiceTTS — thin wrapper around CosyVoice2 inference.

Drop-in for src/audio/tts.py:TTSProcessor at the synthesize() level so
AudioGenerator can swap providers without changing call sites.

Inference path: CosyVoice2 is NOT a diffusers model, it's its own repo
(github.com/FunAudioLLM/CosyVoice). The PyPI `cosyvoice` package is a
broken stub — the only working install path is to clone the upstream
repo and add it (plus its Matcha-TTS submodule) to sys.path. We expect
the clone at <project>/output/external/CosyVoice (override with
COSYVOICE_REPO_PATH). load() raises a clear error if the repo isn't
present, so an uninstalled deployment still boots."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_repo_path() -> Path:
    """Where we look for the cloned CosyVoice repo by default. Lives
    under output/external/ so it's gitignored alongside model caches."""
    # cosyvoice_runtime.py is at src/audio_local/, project root is 2 up.
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "output" / "external" / "CosyVoice"


def _inject_cosyvoice_paths() -> None:
    """Prepend the cloned CosyVoice repo + Matcha-TTS submodule to
    sys.path. Idempotent. Honors COSYVOICE_REPO_PATH env override."""
    repo = Path(os.environ.get("COSYVOICE_REPO_PATH", str(_default_repo_path())))
    if not repo.exists():
        return  # caller will get a clean ImportError below
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (repo, matcha):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# Each preset maps to a bundled reference WAV in the CosyVoice repo's
# asset/ dir, used as the zero-shot/cross-lingual prompt to clone that
# voice's timbre. Adding more presets = bundling more reference clips.
#
# We only ship the 2 references the official repo bundles for now:
#  - zero_shot_prompt.wav → 中文女声 (with known transcript, runs via
#    inference_zero_shot which gives the LLM a text↔token alignment example)
#  - cross_lingual_prompt.wav → 英文男声 (no transcript known, runs via
#    inference_cross_lingual which strips prompt_text from the LLM call)
#
# Notes on what we tried that DIDN'T work:
#  - lucyknada/CosyVoice2-0.5B's spk2info.pt provides 7 voice presets but
#    is v1 SFT data layout-converted to v2 — the LLM still produces
#    coherent-sounding gibberish (no EOS, wrong token distribution) even
#    after schema migration + dtype normalization + transformers downgrade.
#    Our `_migrate_spk2info_v1_to_v2` is left in place as a safety net but
#    inactive on this routing path.
#  - Routing user voice IDs through `inference_cross_lingual(zero_shot_spk_id=...)`
#    against migrated lucyknada spk2info: same gibberish.
#
# UX trade-off: 2 honest voices > 7 voices where 6 don't actually clone.
COSYVOICE2_PRESETS = [
    {"id": "中文女", "name": "中文女声", "gender": "Female", "lang": "zh",
     "ref_wav": "asset/zero_shot_prompt.wav",
     "ref_text": "希望你以后能够做的比我还好呦。",
     "mode": "zero_shot"},
    {"id": "英文男", "name": "English Male", "gender": "Male", "lang": "en",
     "ref_wav": "asset/cross_lingual_prompt.wav",
     "ref_text": "",  # transcript not bundled; cross_lingual mode skips it
     "mode": "cross_lingual"},
]


# HF repo id for the open weights. Configurable via env so users can
# point at a re-uploaded mirror if FunAudioLLM ever takes the original
# down (and so tests can monkeypatch).
DEFAULT_COSYVOICE_HF_ID = os.getenv("LOCAL_TTS_HF_ID", "FunAudioLLM/CosyVoice2-0.5B")


class LocalCosyVoiceTTS:
    """Lazy-loading wrapper around CosyVoice2 inference.

    Two-phase lifecycle:
    - load(): downloads weights (if needed) and constructs the
      cosyvoice.cli.cosyvoice.CosyVoice2 object. Heavy.
    - synthesize(): calls inference_sft for preset-voice TTS.

    The progress_callback hook is plumbed through but not yet driven by
    a real per-step signal — CosyVoice2's inference_sft is generator-
    based, so we update progress on each yielded chunk as a coarse
    indicator (matches what the manager needs for the footer)."""

    def __init__(self, hf_id: str = DEFAULT_COSYVOICE_HF_ID, device: Optional[str] = None):
        self.hf_id = hf_id
        self.device = device or self._pick_device()
        self._cosyvoice: Any = None  # lazy-constructed CosyVoice2 instance
        self._sample_rate: int = 24000  # CosyVoice2 ships 24kHz audio
        # Active label for status() — what voice/text we're currently rendering.
        self.active_label: str = ""

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _model_dir(self) -> Path:
        """Local path where snapshot_download will/did place the weights."""
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            # Fall back to the diffusers / transformers default. We don't
            # construct a path ourselves — let snapshot_download decide.
            return Path()
        return Path(hf_home) / "hub" / ("models--" + self.hf_id.replace("/", "--"))

    def load(self) -> None:
        """Idempotent: download weights if missing, construct the
        CosyVoice2 inference object. Raises with a helpful message if
        the cosyvoice package isn't installed."""
        if self._cosyvoice is not None:
            return

        # The PyPI cosyvoice package is a broken stub; we always import
        # from the cloned upstream repo. Inject its path before importing.
        _inject_cosyvoice_paths()
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        except ImportError as e:
            repo = os.environ.get("COSYVOICE_REPO_PATH", str(_default_repo_path()))
            raise RuntimeError(
                f"CosyVoice repo not importable from {repo!r}. Clone "
                "github.com/FunAudioLLM/CosyVoice (with --recurse-submodules) "
                "to that path, or set COSYVOICE_REPO_PATH. Original error: "
                + str(e)
            ) from e

        from huggingface_hub import snapshot_download

        logger.info(f"[LocalCosyVoiceTTS] downloading {self.hf_id} (if not cached)")
        local_dir = snapshot_download(repo_id=self.hf_id)
        logger.info(f"[LocalCosyVoiceTTS] weights at {local_dir}, constructing CosyVoice2")

        # CosyVoice2 ctor takes the model dir; load_jit/load_trt off by
        # default since we don't ship the export pipeline.
        # fp16=True enables torch.cuda.amp.autocast around the LLM forward
        # pass — needed for the mixed-precision Qwen2 weights to matmul
        # without "mat1 and mat2 must have the same dtype" errors.
        self._cosyvoice = CosyVoice2(
            local_dir, load_jit=False, load_trt=False, fp16=True
        )
        # Some CosyVoice builds expose sample_rate; default to 24k otherwise.
        self._sample_rate = getattr(self._cosyvoice, "sample_rate", 24000)

        # spk2info.pt (downloaded from lucyknada/CosyVoice2-0.5B since the
        # official FunAudioLLM repo doesn't ship one) was saved with the
        # CosyVoice v1 SFT key schema — `embedding`/`speech_token`/`speech_feat`.
        # CosyVoice2's tts() expects v2 keys: `flow_embedding`/`llm_embedding`,
        # `flow_prompt_speech_token`/`llm_prompt_speech_token` (+ `_len`),
        # `prompt_speech_feat` (+ `_len`), `prompt_text` (+ `_len`).
        # Without translation, kwargs fall to defaults (zero-shaped tensors)
        # and inference cascades into a [0, 80] expand error. Translate first,
        # then dtype-normalize the result.
        self._migrate_spk2info_v1_to_v2()
        self._normalize_spk2info_dtype()
        logger.info(
            f"[LocalCosyVoiceTTS] ready (device={self.device}, sr={self._sample_rate}, "
            f"presets={len(self._cosyvoice.frontend.spk2info)})"
        )

    def _migrate_spk2info_v1_to_v2(self) -> None:
        """Translate v1 SFT key schema → v2 zero-shot key schema, in place.

        v1 keys (lucyknada): embedding (1,192), speech_token (1,T), speech_feat (1,F,80)
        v2 keys (what CosyVoice2.tts wants):
          flow_embedding, llm_embedding                     ← from `embedding`
          flow_prompt_speech_token, llm_prompt_speech_token ← from `speech_token`
          flow_prompt_speech_token_len, llm_prompt_speech_token_len ← derived
          prompt_speech_feat                                ← from `speech_feat`
          prompt_speech_feat_len                            ← derived
          prompt_text, prompt_text_len                      ← empty (no text prompt)

        Only acts on entries that look v1-shaped; v2-shaped entries are left
        alone so this stays safe to call against a properly-saved spk2info."""
        import torch
        spk2info = self._cosyvoice.frontend.spk2info
        migrated = 0
        for spk_id, info in list(spk2info.items()):
            if not isinstance(info, dict):
                continue
            if "flow_embedding" in info:
                continue  # already v2
            if "embedding" not in info:
                continue  # unknown shape, skip
            embedding = info["embedding"]
            speech_token = info.get("speech_token")
            speech_feat = info.get("speech_feat")

            # CosyVoice2 has a hard constraint at frontend.py:174-178:
            # speech_feat.shape[1] must equal 2 * speech_token.shape[1].
            # frontend_zero_shot enforces this when computing zero-shot
            # data live, but the spk2info fast path (line 186) doesn't —
            # so lucyknada's raw v1 data (token=507, feat=873) violates
            # the ratio and the LLM/flow consume mismatched lengths,
            # producing intelligible-prosody-but-meaningless babble.
            # Apply the same trim here.
            if speech_token is not None and speech_feat is not None:
                feat_T = speech_feat.shape[1]
                tok_T = speech_token.shape[1]
                token_len = min(feat_T // 2, tok_T)
                speech_token = speech_token[:, :token_len]
                speech_feat = speech_feat[:, : 2 * token_len]

            new_info: Dict[str, Any] = {
                "flow_embedding": embedding,
                "llm_embedding": embedding,
                "prompt_text": torch.zeros(1, 0, dtype=torch.int32),
                "prompt_text_len": torch.zeros(1, dtype=torch.int32),
            }
            if speech_token is not None:
                new_info["flow_prompt_speech_token"] = speech_token
                new_info["llm_prompt_speech_token"] = speech_token
                token_len_t = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
                new_info["flow_prompt_speech_token_len"] = token_len_t
                new_info["llm_prompt_speech_token_len"] = token_len_t
            if speech_feat is not None:
                new_info["prompt_speech_feat"] = speech_feat
                new_info["prompt_speech_feat_len"] = torch.tensor(
                    [speech_feat.shape[1]], dtype=torch.int32
                )
            spk2info[spk_id] = new_info
            migrated += 1
        if migrated:
            logger.info(f"[LocalCosyVoiceTTS] migrated {migrated} spk2info entries v1→v2 schema")

    def _normalize_spk2info_dtype(self) -> None:
        """Cast all float tensors in spk2info to the underlying LLM
        embedding-layer's dtype.

        The llm.model is Qwen2LM, a wrapper around the actual Qwen2
        transformer (`llm.model.llm`) — `next(parameters())` on the
        wrapper may grab a small fp32 helper embedding rather than the
        bf16 transformer weights, giving the wrong target dtype.
        Anchor on the text embedding layer instead, since that's the
        exact layer our speaker embeddings get matmul'd against."""
        import torch
        # Walk the wrapper chain to the underlying transformer's text-embedding
        # layer. CosyVoice2 layout: cosyvoice.model.llm = Qwen2LM (cosyvoice's
        # wrapper); .llm = Qwen2ForCausalLM (HF transformers); .model = Qwen2Model;
        # .embed_tokens = nn.Embedding. Some builds skip the inner .model wrapper,
        # so probe both paths quietly.
        llm_dtype = torch.bfloat16  # safe default for CosyVoice2's Qwen2-0.5B
        for path in (("model", "llm", "llm", "model", "embed_tokens"),
                     ("model", "llm", "llm", "embed_tokens")):
            obj: Any = self._cosyvoice
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                llm_dtype = obj.weight.dtype
                break
            except AttributeError:
                continue
        logger.info(f"[LocalCosyVoiceTTS] normalizing spk2info float tensors → {llm_dtype}")
        if not llm_dtype.is_floating_point:
            return
        casts = 0
        for spk_id, info in self._cosyvoice.frontend.spk2info.items():
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != llm_dtype:
                    info[k] = v.to(llm_dtype)
                    casts += 1
        logger.info(f"[LocalCosyVoiceTTS] cast {casts} spk2info tensors to {llm_dtype}")

    def _preset_for_voice(self, voice: str) -> Dict[str, Any]:
        """Look up the preset dict for `voice`. Falls back to 中文女 if
        the requested voice isn't in our preset list — protects against
        a stale character.voice_id (e.g. someone migrated from cloud and
        kept a `longxxx_v2` ID, or picked a preset we've since dropped)."""
        for p in COSYVOICE2_PRESETS:
            if p["id"] == voice:
                return p
        return COSYVOICE2_PRESETS[0]  # fallback: 中文女

    def _resolve_ref_wav(self, ref_wav_rel: str) -> str:
        """Resolve a preset's `ref_wav` relative path against the cloned
        CosyVoice repo location."""
        repo = Path(os.environ.get("COSYVOICE_REPO_PATH", str(_default_repo_path())))
        return str(repo / ref_wav_rel)

    def list_available_spks(self) -> List[str]:
        """Return preset speaker IDs the loaded model supports. Falls
        back to the static preset list when the model isn't loaded yet
        (so the UI can render the picker before a load fires)."""
        if self._cosyvoice is None:
            return [v["id"] for v in COSYVOICE2_PRESETS]
        try:
            return list(self._cosyvoice.list_available_spks())
        except Exception:
            logger.exception("list_available_spks failed; falling back to static list")
            return [v["id"] for v in COSYVOICE2_PRESETS]

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,  # accepted for API compat; not used by CosyVoice2
        volume: int = 50,         # accepted for API compat; not used by CosyVoice2
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[str, float, str]:
        """Synthesize `text` to `output_path` (.wav). Returns the same
        (path, delay_ms, request_id) tuple shape as the cloud TTS so
        AudioGenerator's call sites don't care which provider answered."""
        import time

        if self._cosyvoice is None:
            self.load()

        voice = voice or "中文女"
        self.active_label = f"{voice}: {text[:24]}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"[LocalCosyVoiceTTS] synthesize voice={voice} chars={len(text)}")

        start = time.time()
        first_package_delay_ms = 0.0
        chunks = []
        # Voice routing: every preset has a bundled reference WAV from
        # the CosyVoice repo's asset/ dir. zero_shot mode is used when we
        # know the reference's transcript (LLM gets a text↔token example
        # for better fidelity); cross_lingual mode for refs without a
        # bundled transcript.
        preset = self._preset_for_voice(voice)
        ref_wav = self._resolve_ref_wav(preset["ref_wav"])
        if preset["mode"] == "zero_shot":
            gen = self._cosyvoice.inference_zero_shot(
                text,
                prompt_text=preset["ref_text"],
                prompt_wav=ref_wav,
                stream=False,
                speed=speech_rate,
            )
        else:  # cross_lingual
            gen = self._cosyvoice.inference_cross_lingual(
                text,
                prompt_wav=ref_wav,
                stream=False,
                speed=speech_rate,
            )
        idx = 0
        for item in gen:
            idx += 1
            if first_package_delay_ms == 0.0:
                first_package_delay_ms = (time.time() - start) * 1000.0
            chunk = self._extract_audio_tensor(item)
            chunks.append(chunk)
            if progress_callback:
                # CosyVoice2 doesn't tell us how many chunks total; report
                # idx/(idx+1) so the bar advances but never claims 100%
                # until we exit the loop.
                try:
                    progress_callback(idx, idx + 1)
                except Exception:
                    logger.exception("progress_callback raised")

        if not chunks:
            raise RuntimeError(
                "CosyVoice2 produced no audio chunks for this input — "
                "voice or text may be invalid"
            )
        self._write_wav(chunks, output_path)
        # Final progress: 100%.
        if progress_callback:
            try:
                progress_callback(idx, idx)
            except Exception:
                logger.exception("progress_callback raised on final tick")

        request_id = f"local-cosyvoice2-{int(start * 1000)}"
        logger.info(
            f"[LocalCosyVoiceTTS] done: chunks={len(chunks)} "
            f"first_chunk={first_package_delay_ms:.0f}ms total={(time.time()-start):.2f}s"
        )
        return output_path, first_package_delay_ms, request_id

    @staticmethod
    def _extract_audio_tensor(item: Any) -> Any:
        """Tolerate the few possible shapes inference_sft yields across
        CosyVoice versions: dict {tts_speech: tensor}, plain tensor, or
        a (sr, tensor) tuple."""
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

        # Concatenate along the time axis. Tensors come in shape (1, T)
        # from CosyVoice; keep that shape for torchaudio.save.
        waveform = torch.cat([self._to_2d(c) for c in chunks], dim=1).cpu()
        # CosyVoice2 emits float32 audio in [-1, 1]. torchaudio.save preserves
        # dtype by default, which writes an IEEE-float WAV (format tag 3).
        # Many consumer players (Windows Media Player, browser <audio>, ffmpeg
        # default decoder paths) garble that into white noise. Force int16 PCM
        # so the output plays correctly everywhere.
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
        """Ensure (channels, samples) shape for torchaudio.save."""
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def unload(self) -> None:
        """Drop the model + free CUDA memory. Idempotent."""
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
