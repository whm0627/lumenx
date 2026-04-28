# STUB — replaced in Task 8 of the implementation plan.
"""LocalWanS2V — placeholder. Real implementation lands in Task 8."""
from __future__ import annotations


class LocalWanS2V:
    """Stub class — Task 8 implements load/generate/unload."""

    def __init__(self, quant: str = "Q4_K_S"):
        self.quant = quant
        self.hf_id = "Wan-AI/Wan2.2-S2V-14B"

    def load(self) -> None:
        raise NotImplementedError("LocalWanS2V.load — implement in Task 8")

    def unload(self) -> None:
        # Safe no-op so tests can patch it freely
        pass

    def generate(self, image: str, audio: str, prompt: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("LocalWanS2V.generate — implement in Task 8")
