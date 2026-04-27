"""LocalLLMConfig — Pydantic settings model for the local LLM runtime."""
from __future__ import annotations

import os
from typing import Dict, Optional

from pydantic import BaseModel, Field

from .vram import QuantMode


class LocalLLMConfig(BaseModel):
    """User-facing configuration for the local LLM runtime."""

    hf_id: str = Field(default="", description="HuggingFace model id, e.g. Qwen/Qwen3-8B-Instruct")
    quant: QuantMode = Field(default=QuantMode.AUTO, description="Quantization mode (transformers path only)")
    idle_seconds: int = Field(default=3, ge=1, le=3600, description="Seconds of idle before unload")
    gguf_file: Optional[str] = Field(
        default=None,
        description="Specific .gguf filename in a GGUF repo; auto-picked if omitted",
    )
    # KV cache window. Default 65536 (64k) accommodates long script processing
    # — full screenplays + asset context + multi-shot polish prompts can easily
    # exceed 32k tokens. KV cache cost depends on model architecture:
    # - Standard transformer (N layers full-attention): cost scales linearly
    #   with n_ctx — beware on dense large models (70B at 64k = several GB KV)
    # - MoE + SSM hybrid (e.g. qwen35moe with 10/40 full-attn layers): SSM
    #   layers have fixed-size state, only the 10 attn layers cost ~1.3GB at 64k
    # Lower if you OOM; raise (up to 131072) for very long contexts.
    n_ctx: int = Field(default=65536, ge=512, le=131072,
                       description="Context window in tokens (KV cache size)")

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        """Hydrate from process environment (loaded from .env at startup)."""
        gguf_file = os.environ.get("LOCAL_LLM_GGUF_FILE", "") or None
        return cls(
            hf_id=os.environ.get("LOCAL_LLM_HF_ID", ""),
            quant=QuantMode(os.environ.get("LOCAL_LLM_QUANT", QuantMode.AUTO.value)),
            idle_seconds=int(os.environ.get("LOCAL_LLM_IDLE_SEC", "3")),
            gguf_file=gguf_file,
            n_ctx=int(os.environ.get("LOCAL_LLM_CTX_SIZE", "65536")),
        )

    def to_env_dict(self) -> Dict[str, str]:
        """Render to env-style dict for persistence via save_user_config()."""
        return {
            "LOCAL_LLM_HF_ID": self.hf_id,
            "LOCAL_LLM_QUANT": self.quant.value,
            "LOCAL_LLM_IDLE_SEC": str(self.idle_seconds),
            "LOCAL_LLM_GGUF_FILE": self.gguf_file or "",
            "LOCAL_LLM_CTX_SIZE": str(self.n_ctx),
        }
