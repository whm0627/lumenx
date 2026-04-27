"""LocalRuntime — synchronous transformers wrapper for one model at a time."""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .vram import QuantMode

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    hf_id: str
    quant_mode: QuantMode
    vram_used_mb: int
    elapsed_sec: float


class LocalRuntime:
    """Owns a single loaded model. Pure sync; ModelManager owns concurrency."""

    def __init__(self, hf_id: str, quant: QuantMode):
        self.hf_id = hf_id
        self.quant = quant
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> LoadResult:
        """Idempotent: re-loading a loaded runtime returns immediately."""
        if self._loaded:
            return self._current_load_result(elapsed_sec=0.0)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.monotonic()
        logger.info(f"Loading {self.hf_id} (quant={self.quant.value})")

        kwargs: Dict = {
            "low_cpu_mem_usage": True,
        }

        if self.quant in (QuantMode.INT4, QuantMode.INT8):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError(
                    "bitsandbytes quantization requires `bitsandbytes` package. "
                    "Run: pip install bitsandbytes>=0.43.0"
                ) from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(self.quant == QuantMode.INT4),
                load_in_8bit=(self.quant == QuantMode.INT8),
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = (
                torch.bfloat16 if self.quant == QuantMode.BF16 else torch.float16
            )
            kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=False)
        if self.tokenizer.chat_template is None:
            raise RuntimeError(
                f"Model {self.hf_id} has no chat_template configured. "
                f"Pick an instruction-tuned model (e.g. *-Instruct variants)."
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)
        self.model.eval()
        self._loaded = True

        elapsed = time.monotonic() - t0
        logger.info(f"Loaded {self.hf_id} in {elapsed:.1f}s")
        return self._current_load_result(elapsed_sec=elapsed)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        # response_format is accepted for interface parity with EmbeddedServerRuntime
        # but transformers has no native JSON-mode; callers requesting strict JSON
        # should use a GGUF model + llama-server instead.
        if not self._loaded:
            raise RuntimeError("Runtime not loaded; call load() first.")
        if response_format is not None:
            logger.warning(
                "response_format requested but LocalRuntime (transformers) does not "
                "support constrained decoding; output may not be valid JSON"
            )

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt — return only newly-generated tokens.
        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        if not self._loaded:
            return
        logger.info(f"Unloading {self.hf_id}")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _current_load_result(self, elapsed_sec: float) -> LoadResult:
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = int(torch.cuda.memory_allocated() // (1024 * 1024))
        return LoadResult(
            hf_id=self.hf_id,
            quant_mode=self.quant,
            vram_used_mb=vram_mb,
            elapsed_sec=elapsed_sec,
        )
