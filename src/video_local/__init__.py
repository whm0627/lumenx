"""Local video generation runtime — Wan2.2-S2V-14B with GGUF quantization.

Mirrors src/audio_local for shape: a manager singleton + state machine
wraps a runtime class that constructs Wan's official WanS2V pipeline,
replacing the DiT's nn.Linear modules with GGUFLinear when a quant
tier (Q8_0 / Q4_K_S) is selected. fp16 mode skips quantization entirely.

This module is import-cheap: Wan and gguf are imported lazily on first
load() so the rest of the app keeps starting if the optional deps or
the cloned Wan repo are missing."""
