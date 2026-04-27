"""Local HuggingFace LLM runtime — model lifecycle management.

Importing this package sets HF_HOME to <project_root>/output/models/LLM/ so
that every transformers / huggingface_hub download lands inside the project.
"""
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LLM_CACHE = _PROJECT_ROOT / "output" / "models" / "LLM"
_LLM_CACHE.mkdir(parents=True, exist_ok=True)

# Honour any preset HF_HOME (e.g., user wants a shared cache); otherwise default to project.
os.environ.setdefault("HF_HOME", str(_LLM_CACHE))
