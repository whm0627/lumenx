# Local LLM Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-managed local HuggingFace LLM runtime to LumenX so users can run script-processing tasks against a locally-loaded model with automatic 3-second-idle unload.

**Architecture:** New `src/llm_local/` package containing a singleton `ModelManager` (asyncio state machine) backed by a `LocalRuntime` (transformers + accelerate + bitsandbytes wrapper). Existing `LLMAdapter` gets a third `local` provider branch that bridges sync→async into the manager. Frontend SettingsPage gains a `LocalLLMPanel` for HF model id, quant, status, and cached-model management. Persistence reuses the existing `save_user_config()` helper.

**Tech Stack:** Python 3.12, FastAPI, asyncio, transformers, accelerate, bitsandbytes, torch (CUDA), pytest, pytest-asyncio, Next.js 14, React 18, TypeScript.

**Reference spec:** [docs/superpowers/specs/2026-04-25-local-llm-runtime-design.md](../specs/2026-04-25-local-llm-runtime-design.md)

---

## Setup Notes

- All Python paths assume the project root `s:\AI\lumenxDev\lumenx`.
- The Python venv lives at `.venv/` and is created with Python 3.12.9.
- Backend dev server runs via `npm run dev` with `--reload` enabled — file changes auto-restart uvicorn. Frontend has HMR.
- Existing testpaths config (`pyproject.toml`) uses `tests/`. New tests go under `tests/llm_local/`.
- `bash` shell is the default; PowerShell is also available. Examples use bash syntax.
- Commits use Conventional Commits prefixes (`feat:`, `test:`, `chore:`, `fix:`).

---

## File Structure

**New files:**

```
src/llm_local/
├── __init__.py          # package marker; sets HF_HOME to project-local path
├── vram.py              # VRAM detection + auto quant strategy
├── config.py            # Pydantic settings model + .env load/save bridge
├── runtime.py           # LocalRuntime: transformers wrapper (load/chat/unload)
├── manager.py           # ModelManager: singleton, state machine, idle watcher
└── api.py               # FastAPI APIRouter with /llm/local/* endpoints

tests/llm_local/
├── __init__.py
├── test_vram.py         # VRAM/quant logic (mocked torch.cuda)
├── test_config.py       # Settings persistence
├── test_manager.py      # State machine + lock + idle watcher (mocked runtime)
└── test_api.py          # FastAPI endpoints (TestClient + mocked manager)

frontend/src/components/settings/
└── LocalLLMPanel.tsx    # Local LLM configuration + status panel
```

**Modified files:**

```
requirements.txt                                    # add torch/transformers/bnb deps
src/apps/comic_gen/llm_adapter.py                   # add 'local' provider branch
src/apps/comic_gen/api.py                           # set HF_HOME, include router, capture loop
src/apps/comic_gen/llm.py                           # strip response_format for local provider
frontend/src/lib/api.ts                             # client functions for /llm/local/*
frontend/src/components/settings/SettingsPage.tsx   # add LLM Provider section + embed panel
.env.example                                        # document new LOCAL_LLM_* keys
```

---

## Task 1: Add Python dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add LLM runtime dependencies to requirements.txt**

Append the following block to [requirements.txt](../../requirements.txt) (after the existing `# Cloud Services` section, before the `# Optional Dependencies` comment):

```
# Local LLM Runtime
torch>=2.3.0
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.43.0
huggingface_hub>=0.25.0
sentencepiece>=0.2.0
openai>=1.0.0
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

(`openai` is needed for the existing `LLMAdapter.openai` branch when users actually use the openai provider; we add it now while we're touching deps. `pytest` and `pytest-asyncio` enable the new tests.)

- [ ] **Step 2: Install the dependencies**

Run from `s:\AI\lumenxDev\lumenx`:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Expected: pip downloads torch (~2 GB), transformers, bitsandbytes, etc. Total install ~5–7 GB. Final line: `Successfully installed ...`.

If torch install fails because of missing CUDA wheel, install the CUDA 12.1 build explicitly:

```bash
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- [ ] **Step 3: Verify torch sees the GPU**

```bash
.venv/Scripts/python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected: `CUDA: True Device: NVIDIA GeForce RTX 4090` (or whatever GPU is installed).

If `CUDA: False` → stop and report. The local LLM feature requires CUDA.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add local LLM runtime dependencies (torch, transformers, bitsandbytes)"
```

---

## Task 2: VRAM detection and quant strategy

**Files:**
- Create: `src/llm_local/__init__.py`
- Create: `src/llm_local/vram.py`
- Create: `tests/llm_local/__init__.py`
- Create: `tests/llm_local/test_vram.py`

- [ ] **Step 1: Create the package marker**

Create [src/llm_local/__init__.py](../../src/llm_local/__init__.py) with this content:

```python
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
```

- [ ] **Step 2: Create the test package marker**

Create [tests/llm_local/__init__.py](../../tests/llm_local/__init__.py) as an empty file:

```python
```

- [ ] **Step 3: Write the failing test for VRAM detection and quant strategy**

Create [tests/llm_local/test_vram.py](../../tests/llm_local/test_vram.py):

```python
"""Tests for VRAM detection and auto quant strategy."""
from unittest.mock import patch

import pytest

from src.llm_local.vram import (
    QuantMode,
    detect_vram_total_mb,
    pick_quant_for_model,
    estimate_model_size_b,
)


class TestDetectVram:
    def test_returns_total_mb_when_cuda_available(self):
        with patch("src.llm_local.vram.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3
            assert detect_vram_total_mb() == 24 * 1024

    def test_returns_zero_when_no_cuda(self):
        with patch("src.llm_local.vram.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            assert detect_vram_total_mb() == 0


class TestPickQuant:
    def test_4b_picks_bf16(self):
        assert pick_quant_for_model(params_b=4.0, vram_mb=24 * 1024) == QuantMode.BF16

    def test_8b_picks_bf16(self):
        assert pick_quant_for_model(params_b=8.0, vram_mb=24 * 1024) == QuantMode.BF16

    def test_14b_picks_8bit(self):
        assert pick_quant_for_model(params_b=14.0, vram_mb=24 * 1024) == QuantMode.INT8

    def test_32b_picks_4bit(self):
        assert pick_quant_for_model(params_b=32.0, vram_mb=24 * 1024) == QuantMode.INT4

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match="too large"):
            pick_quant_for_model(params_b=70.0, vram_mb=24 * 1024)

    def test_low_vram_8b_picks_4bit(self):
        # 8B in bf16 needs ~16GB; with only 12GB VRAM, fall back to 4bit
        assert pick_quant_for_model(params_b=8.0, vram_mb=12 * 1024) == QuantMode.INT4


class TestEstimateModelSize:
    def test_extracts_from_hf_id_with_b_suffix(self):
        assert estimate_model_size_b("Qwen/Qwen3-8B-Instruct") == 8.0
        assert estimate_model_size_b("Qwen/Qwen3-14B") == 14.0
        assert estimate_model_size_b("meta-llama/Llama-3-70B-Instruct") == 70.0

    def test_extracts_from_decimal_b(self):
        assert estimate_model_size_b("Qwen/Qwen3-1.7B") == 1.7

    def test_extracts_0_5b(self):
        assert estimate_model_size_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5

    def test_returns_none_when_no_size_in_id(self):
        assert estimate_model_size_b("gpt2") is None
```

- [ ] **Step 4: Run the test to confirm it fails**

```bash
cd /s/AI/lumenxDev/lumenx && .venv/Scripts/python.exe -m pytest tests/llm_local/test_vram.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.llm_local.vram'` or similar import failure.

- [ ] **Step 5: Implement the VRAM module**

Create [src/llm_local/vram.py](../../src/llm_local/vram.py):

```python
"""VRAM detection and automatic quantization selection."""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class QuantMode(str, Enum):
    AUTO = "auto"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "8bit"
    INT4 = "4bit"


# Approximate VRAM (MB) needed per billion parameters at each precision.
# These are inference estimates including KV cache headroom (~20%).
_VRAM_PER_B = {
    QuantMode.FP16: 2400,
    QuantMode.BF16: 2400,
    QuantMode.INT8: 1300,
    QuantMode.INT4: 700,
}


def detect_vram_total_mb() -> int:
    """Return total VRAM of device 0 in MiB, or 0 if no CUDA GPU."""
    if torch is None or not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory // (1024 * 1024))


def estimate_model_size_b(hf_id: str) -> Optional[float]:
    """Best-effort: parse the parameter count (in billions) out of a HF model id.

    Recognises forms like 'Qwen3-8B', 'Qwen3-1.7B', 'Qwen2.5-0.5B-Instruct'.
    Returns None if no '<number>B' token is found (caller should fetch HF config
    to determine size).
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:[\W_]|$)", hf_id)
    return float(match.group(1)) if match else None


def pick_quant_for_model(params_b: float, vram_mb: int) -> QuantMode:
    """Pick the highest-precision quant that fits comfortably in VRAM.

    Defaults from spec section 5.3:
      ≤ 4B   → bf16
      7-8B   → bf16
      13-14B → 8bit
      32B    → 4bit
      > ~50B → reject

    If the spec default doesn't fit available VRAM, step down until it does.
    """
    if params_b <= 8:
        candidates = [QuantMode.BF16, QuantMode.INT8, QuantMode.INT4]
    elif params_b <= 16:
        candidates = [QuantMode.INT8, QuantMode.INT4]
    elif params_b <= 35:
        candidates = [QuantMode.INT4]
    else:
        raise ValueError(
            f"Model size {params_b}B is too large for the supported quant range. "
            f"Pick a smaller model or use external inference."
        )

    for q in candidates:
        needed = int(_VRAM_PER_B[q] * params_b)
        if needed <= vram_mb:
            return q

    raise ValueError(
        f"Model size {params_b}B does not fit in {vram_mb}MB VRAM at any supported quant."
    )
```

- [ ] **Step 6: Run the tests to confirm they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_vram.py -v
```

Expected: all tests pass (`11 passed`).

- [ ] **Step 7: Commit**

```bash
git add src/llm_local/__init__.py src/llm_local/vram.py tests/llm_local/__init__.py tests/llm_local/test_vram.py
git commit -m "feat(llm_local): add VRAM detection and quant auto-selection"
```

---

## Task 3: Config module (Pydantic settings + persistence)

**Files:**
- Create: `src/llm_local/config.py`
- Create: `tests/llm_local/test_config.py`

- [ ] **Step 1: Write the failing test**

Create [tests/llm_local/test_config.py](../../tests/llm_local/test_config.py):

```python
"""Tests for LocalLLMConfig persistence."""
from unittest.mock import patch

from src.llm_local.config import LocalLLMConfig
from src.llm_local.vram import QuantMode


class TestLocalLLMConfig:
    def test_defaults(self):
        cfg = LocalLLMConfig()
        assert cfg.hf_id == ""
        assert cfg.quant == QuantMode.AUTO
        assert cfg.idle_seconds == 3

    def test_from_env(self):
        env = {
            "LOCAL_LLM_HF_ID": "Qwen/Qwen3-8B-Instruct",
            "LOCAL_LLM_QUANT": "bf16",
            "LOCAL_LLM_IDLE_SEC": "10",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.hf_id == "Qwen/Qwen3-8B-Instruct"
        assert cfg.quant == QuantMode.BF16
        assert cfg.idle_seconds == 10

    def test_from_env_missing_uses_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = LocalLLMConfig.from_env()
        assert cfg.hf_id == ""
        assert cfg.quant == QuantMode.AUTO
        assert cfg.idle_seconds == 3

    def test_to_env_dict(self):
        cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-4B", quant=QuantMode.INT8, idle_seconds=5)
        d = cfg.to_env_dict()
        assert d == {
            "LOCAL_LLM_HF_ID": "Qwen/Qwen3-4B",
            "LOCAL_LLM_QUANT": "8bit",
            "LOCAL_LLM_IDLE_SEC": "5",
        }
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_config.py -v
```

Expected: import error.

- [ ] **Step 3: Implement the config module**

Create [src/llm_local/config.py](../../src/llm_local/config.py):

```python
"""LocalLLMConfig — Pydantic settings model for the local LLM runtime."""
from __future__ import annotations

import os
from typing import Dict

from pydantic import BaseModel, Field

from .vram import QuantMode


class LocalLLMConfig(BaseModel):
    """User-facing configuration for the local LLM runtime."""

    hf_id: str = Field(default="", description="HuggingFace model id, e.g. Qwen/Qwen3-8B-Instruct")
    quant: QuantMode = Field(default=QuantMode.AUTO, description="Quantization mode")
    idle_seconds: int = Field(default=3, ge=1, le=3600, description="Seconds of idle before unload")

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        """Hydrate from process environment (loaded from .env at startup)."""
        return cls(
            hf_id=os.environ.get("LOCAL_LLM_HF_ID", ""),
            quant=QuantMode(os.environ.get("LOCAL_LLM_QUANT", QuantMode.AUTO.value)),
            idle_seconds=int(os.environ.get("LOCAL_LLM_IDLE_SEC", "3")),
        )

    def to_env_dict(self) -> Dict[str, str]:
        """Render to env-style dict for persistence via save_user_config()."""
        return {
            "LOCAL_LLM_HF_ID": self.hf_id,
            "LOCAL_LLM_QUANT": self.quant.value,
            "LOCAL_LLM_IDLE_SEC": str(self.idle_seconds),
        }
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/config.py tests/llm_local/test_config.py
git commit -m "feat(llm_local): add LocalLLMConfig with .env persistence"
```

---

## Task 4: LocalRuntime (transformers wrapper)

This is the only module that imports `transformers` directly. We test it with a tiny real model (`hf-internal-testing/tiny-random-Qwen2ForCausalLM` or `Qwen/Qwen2.5-0.5B-Instruct` if smaller is unavailable). The test downloads ~50 MB once.

**Files:**
- Create: `src/llm_local/runtime.py`
- Create: `tests/llm_local/test_runtime.py`

- [ ] **Step 1: Write the failing integration test**

Create [tests/llm_local/test_runtime.py](../../tests/llm_local/test_runtime.py):

```python
"""Integration tests for LocalRuntime using a tiny real model.

Marked as `slow` because it downloads weights on first run. Skipped when no GPU.
"""
import pytest
import torch

from src.llm_local.runtime import LocalRuntime, LoadResult
from src.llm_local.vram import QuantMode


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LocalRuntime requires CUDA GPU",
)

# A tiny instruct model that's quick to download (~1 GB).
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def runtime():
    rt = LocalRuntime(hf_id=TEST_MODEL, quant=QuantMode.BF16)
    yield rt
    rt.unload()


class TestLocalRuntime:
    def test_load_returns_load_result(self, runtime):
        result = runtime.load()
        assert isinstance(result, LoadResult)
        assert result.hf_id == TEST_MODEL
        assert result.vram_used_mb > 0

    def test_chat_returns_non_empty_string(self, runtime):
        runtime.load()
        out = runtime.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)
        assert isinstance(out, str)
        assert len(out.strip()) > 0

    def test_chat_strips_input_prompt_from_output(self, runtime):
        runtime.load()
        out = runtime.chat(
            [{"role": "user", "content": "Reply with exactly the word: TESTOUTPUT"}],
            max_new_tokens=8,
        )
        # The output must not contain the user's instruction text verbatim
        assert "Reply with exactly" not in out

    def test_unload_frees_vram(self, runtime):
        runtime.load()
        before = torch.cuda.memory_allocated()
        runtime.unload()
        after = torch.cuda.memory_allocated()
        assert after < before
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_runtime.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.llm_local.runtime'`.

- [ ] **Step 3: Implement LocalRuntime**

Create [src/llm_local/runtime.py](../../src/llm_local/runtime.py):

```python
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
    ) -> str:
        if not self._loaded:
            raise RuntimeError("Runtime not loaded; call load() first.")

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
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_runtime.py -v -s
```

Expected: 4 passed (first run takes ~2 min for the download). If GPU not present, all skipped.

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/runtime.py tests/llm_local/test_runtime.py
git commit -m "feat(llm_local): add LocalRuntime transformers wrapper with chat_template support"
```

---

## Task 5: ModelManager (state machine + lock + idle watcher)

**Files:**
- Create: `src/llm_local/manager.py`
- Create: `tests/llm_local/test_manager.py`

- [ ] **Step 1: Write the failing test**

Create [tests/llm_local/test_manager.py](../../tests/llm_local/test_manager.py):

```python
"""Tests for ModelManager state machine using a stub runtime."""
import asyncio
from unittest.mock import MagicMock

import pytest

from src.llm_local.config import LocalLLMConfig
from src.llm_local.manager import ModelManager, ModelState
from src.llm_local.runtime import LoadResult
from src.llm_local.vram import QuantMode


def make_stub_runtime_factory(load_result: LoadResult, chat_response: str = "OK"):
    """Build a factory that returns a fresh MagicMock runtime each call."""
    def factory(hf_id, quant):
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = load_result
        rt.chat.return_value = chat_response
        return rt
    return factory


@pytest.fixture
def stub_load_result():
    return LoadResult(
        hf_id="Qwen/Qwen3-8B-Instruct",
        quant_mode=QuantMode.BF16,
        vram_used_mb=15000,
        elapsed_sec=30.0,
    )


@pytest.fixture
def manager(stub_load_result):
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    factory = make_stub_runtime_factory(stub_load_result)
    return ModelManager(config=cfg, runtime_factory=factory)


@pytest.mark.asyncio
async def test_initial_state_is_unloaded(manager):
    status = manager.status()
    assert status["state"] == ModelState.UNLOADED.value


@pytest.mark.asyncio
async def test_chat_lazy_loads(manager):
    out = await manager.chat([{"role": "user", "content": "hi"}])
    assert out == "OK"
    assert manager.status()["state"] == ModelState.READY.value


@pytest.mark.asyncio
async def test_unload_returns_to_unloaded(manager):
    await manager.chat([{"role": "user", "content": "hi"}])
    await manager.unload()
    assert manager.status()["state"] == ModelState.UNLOADED.value


@pytest.mark.asyncio
async def test_chat_with_empty_hf_id_raises(stub_load_result):
    cfg = LocalLLMConfig(hf_id="", quant=QuantMode.AUTO, idle_seconds=3)
    factory = make_stub_runtime_factory(stub_load_result)
    mgr = ModelManager(config=cfg, runtime_factory=factory)
    with pytest.raises(RuntimeError, match="Configure local LLM first"):
        await mgr.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_configure_with_different_id_unloads_current(manager, stub_load_result):
    await manager.chat([{"role": "user", "content": "hi"}])
    assert manager.status()["state"] == ModelState.READY.value
    new_cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-4B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    await manager.configure(new_cfg)
    # Should be unloaded after switching id
    assert manager.status()["state"] == ModelState.UNLOADED.value
    assert manager.status()["hf_id"] == "Qwen/Qwen3-4B-Instruct"


@pytest.mark.asyncio
async def test_idle_watcher_unloads_after_timeout(stub_load_result):
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=1)
    factory = make_stub_runtime_factory(stub_load_result)
    mgr = ModelManager(config=cfg, runtime_factory=factory, watcher_interval_sec=0.2)
    await mgr.start_idle_watcher()
    try:
        await mgr.chat([{"role": "user", "content": "hi"}])
        assert mgr.status()["state"] == ModelState.READY.value
        # Wait long enough for the watcher to trigger
        await asyncio.sleep(2.0)
        assert mgr.status()["state"] == ModelState.UNLOADED.value
    finally:
        await mgr.stop_idle_watcher()


@pytest.mark.asyncio
async def test_concurrent_chat_serialised_by_lock(manager):
    """Two chats started in parallel should both succeed (serialised by lock)."""
    results = await asyncio.gather(
        manager.chat([{"role": "user", "content": "first"}]),
        manager.chat([{"role": "user", "content": "second"}]),
    )
    assert results == ["OK", "OK"]


@pytest.mark.asyncio
async def test_load_oom_steps_down_quant(stub_load_result):
    """OOM at bf16 should auto-step-down to 8bit."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)

    # Track which quant each call used
    call_quants = []

    def factory(hf_id, quant):
        call_quants.append(quant)
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        if quant == QuantMode.BF16:
            rt.load.side_effect = RuntimeError("CUDA out of memory")
        else:
            rt.load.return_value = stub_load_result
            rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    out = await mgr.chat([{"role": "user", "content": "hi"}])
    assert out == "OK"
    assert QuantMode.BF16 in call_quants  # tried bf16 first
    assert QuantMode.INT8 in call_quants  # stepped down to 8bit


@pytest.mark.asyncio
async def test_load_non_oom_error_does_not_step_down(stub_load_result):
    """Non-OOM errors propagate immediately, no step-down."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    call_quants = []

    def factory(hf_id, quant):
        call_quants.append(quant)
        rt = MagicMock()
        rt.load.side_effect = RuntimeError("model not found")
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    with pytest.raises(RuntimeError, match="model not found"):
        await mgr.chat([{"role": "user", "content": "hi"}])
    assert call_quants == [QuantMode.BF16]  # only tried once
```

Also add to [tests/llm_local/__init__.py](../../tests/llm_local/__init__.py) — leave empty, but ensure pytest-asyncio is configured. Add an asyncio marker config to [pyproject.toml](../../pyproject.toml) if not already there.

- [ ] **Step 2: Configure pytest-asyncio**

Edit [pyproject.toml](../../pyproject.toml) to add the asyncio_mode config in the `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
asyncio_mode = "auto"
```

(Add only the `asyncio_mode = "auto"` line; keep the existing config.)

- [ ] **Step 3: Run the test to confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_manager.py -v
```

Expected: import error for `src.llm_local.manager`.

- [ ] **Step 4: Implement ModelManager**

Create [src/llm_local/manager.py](../../src/llm_local/manager.py):

```python
"""ModelManager — singleton, state machine, lock, idle watcher."""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

from .config import LocalLLMConfig
from .runtime import LocalRuntime
from .vram import QuantMode, detect_vram_total_mb, estimate_model_size_b, pick_quant_for_model

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"  # not currently distinguished from LOADING; reserved
    LOADING = "LOADING"
    READY = "READY"
    ERROR = "ERROR"


RuntimeFactory = Callable[[str, QuantMode], LocalRuntime]


def _default_runtime_factory(hf_id: str, quant: QuantMode) -> LocalRuntime:
    return LocalRuntime(hf_id=hf_id, quant=quant)


class ModelManager:
    """Asyncio singleton managing one LocalRuntime at a time."""

    _instance: Optional["ModelManager"] = None

    def __init__(
        self,
        config: LocalLLMConfig,
        runtime_factory: RuntimeFactory = _default_runtime_factory,
        watcher_interval_sec: float = 1.0,
    ):
        self.config = config
        self._runtime_factory = runtime_factory
        self._watcher_interval = watcher_interval_sec
        self._runtime: Optional[LocalRuntime] = None
        self._state = ModelState.UNLOADED
        self._error_msg: Optional[str] = None
        self._last_used_ts = time.time()
        self._lock = asyncio.Lock()
        self._watcher_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ---- singleton accessor used by LLMAdapter ----

    @classmethod
    def install(cls, config: LocalLLMConfig) -> "ModelManager":
        cls._instance = ModelManager(config=config)
        return cls._instance

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            raise RuntimeError("ModelManager not installed; call install() at startup.")
        return cls._instance

    # ---- public async API ----

    async def configure(self, new_config: LocalLLMConfig) -> None:
        async with self._lock:
            id_changed = new_config.hf_id != self.config.hf_id
            quant_changed = new_config.quant != self.config.quant
            self.config = new_config
            if (id_changed or quant_changed) and self._state == ModelState.READY:
                await self._unload_locked()

    async def load(self) -> Dict:
        async with self._lock:
            await self._load_locked()
            return self.status()

    async def unload(self) -> None:
        async with self._lock:
            await self._unload_locked()

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        async with self._lock:
            if not self.config.hf_id:
                raise RuntimeError(
                    "Configure local LLM first (set HF model id in Settings)."
                )
            if self._state != ModelState.READY:
                await self._load_locked()
            assert self._runtime is not None
            self._last_used_ts = time.time()
            try:
                # Run sync transformers call off the event loop
                response = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self._runtime.chat(messages, **kwargs)
                )
                self._last_used_ts = time.time()
                return response
            except Exception as e:
                self._state = ModelState.ERROR
                self._error_msg = str(e)
                logger.exception("chat() failed")
                raise

    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Sync bridge for callers running outside the FastAPI event loop."""
        if self._loop is None:
            raise RuntimeError(
                "ModelManager event loop not captured; call capture_loop() at startup."
            )
        future = asyncio.run_coroutine_threadsafe(self.chat(messages, **kwargs), self._loop)
        return future.result()

    def status(self) -> Dict:
        return {
            "state": self._state.value,
            "hf_id": self.config.hf_id,
            "quant_mode": self._effective_quant_label(),
            "vram_used_mb": self._runtime._current_load_result(0).vram_used_mb if self._runtime and self._runtime._loaded else 0,
            "vram_total_mb": detect_vram_total_mb(),
            "last_used_ts": self._last_used_ts,
            "idle_seconds": self.config.idle_seconds,
            "error": self._error_msg,
        }

    # ---- idle watcher ----

    def capture_loop(self) -> None:
        """Capture the running event loop for the sync bridge. Call at app startup."""
        self._loop = asyncio.get_running_loop()

    async def start_idle_watcher(self) -> None:
        if self._watcher_task is not None:
            return
        self._watcher_task = asyncio.create_task(self._idle_watcher_loop())

    async def stop_idle_watcher(self) -> None:
        if self._watcher_task is None:
            return
        self._watcher_task.cancel()
        try:
            await self._watcher_task
        except asyncio.CancelledError:
            pass
        self._watcher_task = None

    async def _idle_watcher_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._watcher_interval)
                if self._state != ModelState.READY:
                    continue
                if time.time() - self._last_used_ts > self.config.idle_seconds:
                    logger.info("Idle threshold exceeded; unloading model")
                    async with self._lock:
                        # Re-check after acquiring lock (someone may have used it)
                        if (
                            self._state == ModelState.READY
                            and time.time() - self._last_used_ts > self.config.idle_seconds
                        ):
                            await self._unload_locked()
        except asyncio.CancelledError:
            raise

    # ---- internals (assume lock held) ----

    # Quant step-down sequence used when an OOM is hit during load (spec §9).
    _OOM_STEP_DOWN = {
        QuantMode.FP16: QuantMode.INT8,
        QuantMode.BF16: QuantMode.INT8,
        QuantMode.INT8: QuantMode.INT4,
        QuantMode.INT4: None,
    }

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg or exc.__class__.__name__ == "OutOfMemoryError"

    async def _load_locked(self) -> None:
        if self._state == ModelState.READY and self._runtime is not None:
            return  # idempotent
        self._error_msg = None
        self._state = ModelState.LOADING

        # Resolve quant if AUTO
        quant = self.config.quant
        if quant == QuantMode.AUTO:
            params_b = estimate_model_size_b(self.config.hf_id)
            vram_mb = detect_vram_total_mb()
            if params_b is None:
                quant = QuantMode.BF16  # safe default, fallback if not parseable
                logger.warning(f"Could not infer size from {self.config.hf_id}; defaulting to bf16")
            else:
                quant = pick_quant_for_model(params_b=params_b, vram_mb=vram_mb)
                logger.info(f"Auto-picked quant={quant.value} for {params_b}B in {vram_mb}MB VRAM")

        attempt_quant = quant
        while attempt_quant is not None:
            try:
                self._runtime = self._runtime_factory(self.config.hf_id, attempt_quant)
                await asyncio.get_running_loop().run_in_executor(None, self._runtime.load)
                self._state = ModelState.READY
                self._last_used_ts = time.time()
                if attempt_quant != quant:
                    logger.warning(f"Loaded at fallback quant={attempt_quant.value} after OOM at {quant.value}")
                return
            except Exception as e:
                # Clean up any partial runtime
                if self._runtime is not None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
                    except Exception:
                        pass
                    self._runtime = None
                if self._is_oom_error(e):
                    next_quant = self._OOM_STEP_DOWN.get(attempt_quant)
                    if next_quant is not None:
                        logger.warning(f"OOM at {attempt_quant.value}; stepping down to {next_quant.value}")
                        attempt_quant = next_quant
                        continue
                # Non-OOM or no further step-down → fail
                self._state = ModelState.ERROR
                self._error_msg = str(e)
                logger.exception("load() failed")
                raise

    async def _unload_locked(self) -> None:
        if self._runtime is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._runtime.unload)
            self._runtime = None
        self._state = ModelState.UNLOADED
        self._error_msg = None

    def _effective_quant_label(self) -> str:
        if self._runtime is not None:
            return self._runtime.quant.value
        return self.config.quant.value
```

- [ ] **Step 5: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_manager.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/llm_local/manager.py tests/llm_local/test_manager.py pyproject.toml
git commit -m "feat(llm_local): add ModelManager with state machine, lock, and idle watcher"
```

---

## Task 6: FastAPI router for /llm/local/*

**Files:**
- Create: `src/llm_local/api.py`
- Create: `tests/llm_local/test_api.py`

- [ ] **Step 1: Write the failing test**

Create [tests/llm_local/test_api.py](../../tests/llm_local/test_api.py):

```python
"""Tests for /llm/local/* FastAPI router using a stub manager."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.llm_local.api import router
from src.llm_local.manager import ModelManager, ModelState


@pytest.fixture
def app_with_router():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def stub_manager():
    mgr = MagicMock(spec=ModelManager)
    mgr.status.return_value = {
        "state": ModelState.UNLOADED.value,
        "hf_id": "Qwen/Qwen3-8B-Instruct",
        "quant_mode": "auto",
        "vram_used_mb": 0,
        "vram_total_mb": 24576,
        "last_used_ts": 0.0,
        "idle_seconds": 3,
        "error": None,
    }
    mgr.configure = AsyncMock(return_value=None)
    mgr.load = AsyncMock(return_value=mgr.status.return_value)
    mgr.unload = AsyncMock(return_value=None)
    mgr.chat = AsyncMock(return_value="Hello")
    return mgr


@pytest.fixture
def client(app_with_router, stub_manager):
    with patch("src.llm_local.api.ModelManager.get", return_value=stub_manager):
        yield TestClient(app_with_router)


class TestApi:
    def test_status(self, client):
        resp = client.get("/llm/local/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "UNLOADED"
        assert body["vram_total_mb"] == 24576

    def test_configure(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 3}
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "persisted": True}
        mock_save.assert_called_once()

    def test_configure_validates_idle_seconds(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 0}
        resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 422  # Pydantic ge=1

    def test_load(self, client):
        resp = client.post("/llm/local/load")
        assert resp.status_code == 200
        assert resp.json()["state"] == "UNLOADED"  # the stub status

    def test_unload(self, client):
        resp = client.post("/llm/local/unload")
        assert resp.status_code == 200
        assert resp.json()["state"] == "UNLOADED"

    def test_test_endpoint(self, client):
        resp = client.post("/llm/local/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["response"] == "Hello"

    def test_cached_lists_models(self, client, tmp_path, monkeypatch):
        # Build a fake HF cache layout
        cache = tmp_path / "models--Qwen--Qwen3-8B-Instruct" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "config.json").write_bytes(b"x" * 1000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "Qwen/Qwen3-8B-Instruct"
        assert body[0]["size_bytes"] == 1000
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_api.py -v
```

Expected: import error.

- [ ] **Step 3: Implement the router**

Create [src/llm_local/api.py](../../src/llm_local/api.py):

```python
"""FastAPI router exposing /llm/local/* endpoints."""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .config import LocalLLMConfig
from .manager import ModelManager
from .vram import QuantMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm/local", tags=["local-llm"])

# Resolved HF cache directory (set in __init__.py via HF_HOME).
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


# ---- request / response models ----

class ConfigurePayload(BaseModel):
    hf_id: str = Field(..., min_length=1)
    quant: QuantMode = QuantMode.AUTO
    idle_seconds: int = Field(default=3, ge=1, le=3600)


class CachedModelInfo(BaseModel):
    hf_id: str
    size_bytes: int
    snapshot_path: str


# ---- endpoints ----

@router.get("/status")
def get_status() -> dict:
    return ModelManager.get().status()


@router.post("/configure")
async def configure(payload: ConfigurePayload) -> dict:
    # Defer the import to avoid circular imports during testing
    from src.apps.comic_gen.api import save_user_config  # type: ignore

    cfg = LocalLLMConfig(
        hf_id=payload.hf_id, quant=payload.quant, idle_seconds=payload.idle_seconds
    )
    save_user_config({"LLM_PROVIDER": "local", **cfg.to_env_dict()})
    # Reflect into process env so subsequent reads see new values immediately
    os.environ["LLM_PROVIDER"] = "local"
    for k, v in cfg.to_env_dict().items():
        os.environ[k] = v
    await ModelManager.get().configure(cfg)
    return {"ok": True, "persisted": True}


@router.post("/load")
async def load() -> dict:
    try:
        return await ModelManager.get().load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload() -> dict:
    await ModelManager.get().unload()
    return ModelManager.get().status()


@router.post("/test")
async def test_chat() -> dict:
    t0 = time.monotonic()
    try:
        response = await ModelManager.get().chat(
            [{"role": "user", "content": "Say hello in one word."}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": bool(response.strip()), "elapsed_sec": round(time.monotonic() - t0, 2), "response": response}


@router.get("/cached")
def list_cached() -> List[CachedModelInfo]:
    """Scan HF cache layout `models--<org>--<name>/snapshots/<rev>` and report sizes."""
    if not HF_CACHE_DIR.exists():
        return []
    out: List[CachedModelInfo] = []
    for entry in HF_CACHE_DIR.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        # 'models--Qwen--Qwen3-8B-Instruct' → 'Qwen/Qwen3-8B-Instruct'
        parts = entry.name[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        hf_id = "/".join(parts)
        snapshots_dir = entry / "snapshots"
        if not snapshots_dir.exists():
            continue
        # Take the largest snapshot (latest)
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        size = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())
        out.append(CachedModelInfo(hf_id=hf_id, size_bytes=size, snapshot_path=str(latest)))
    return out


@router.delete("/cached")
def delete_cached(hf_id: str = Query(..., description="HF model id, e.g. Qwen/Qwen3-8B-Instruct")) -> dict:
    """Delete a cached snapshot. Refuses if the model is currently loaded."""
    mgr = ModelManager.get()
    status = mgr.status()
    if status["hf_id"] == hf_id and status["state"] == "READY":
        raise HTTPException(
            status_code=409,
            detail=f"Model {hf_id} is currently loaded. Unload first before deleting.",
        )
    cache_subdir = HF_CACHE_DIR / f"models--{hf_id.replace('/', '--')}"
    if not cache_subdir.exists():
        raise HTTPException(status_code=404, detail=f"No cached snapshot found for {hf_id}")
    shutil.rmtree(cache_subdir)
    return {"ok": True, "deleted": str(cache_subdir)}
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_api.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/api.py tests/llm_local/test_api.py
git commit -m "feat(llm_local): add /llm/local/* FastAPI router"
```

---

## Task 7: Wire LLMAdapter "local" branch

**Files:**
- Modify: `src/apps/comic_gen/llm_adapter.py`

- [ ] **Step 1: Add the local branch to chat()**

Edit [src/apps/comic_gen/llm_adapter.py](../../src/apps/comic_gen/llm_adapter.py):

Change `chat()` body to dispatch to ModelManager when provider is `local`. Replace the existing `chat()` method (lines 66–101) with:

```python
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send a chat completion request and return the response content.

        For provider='local' the call is dispatched to the in-process ModelManager.
        response_format is ignored for local (no native JSON mode); rely on prompt instructions.
        """
        if self.provider == "local":
            from src.llm_local.manager import ModelManager
            return ModelManager.get().chat_sync(messages)

        client = self._get_client()
        model = model or self._get_default_model()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            provider_label = "DashScope" if self.provider != "openai" else "OpenAI"
            raise RuntimeError(f"{provider_label} API error: {e}") from e
```

Also update `is_configured` (lines 32–36) to handle the local provider:

```python
    @property
    def is_configured(self) -> bool:
        if self.provider == "local":
            return bool(os.getenv("LOCAL_LLM_HF_ID"))
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        return bool(os.getenv("DASHSCOPE_API_KEY"))
```

And update the module docstring at the top of the file (lines 1–14) to mention three providers:

```python
"""
LLM Adapter - Unified interface for DashScope, OpenAI-compatible APIs, and local HF models.

Supports three providers:
  - dashscope (default): Alibaba Cloud DashScope via OpenAI-compatible endpoint
  - openai: Any OpenAI-compatible API (OpenAI, DeepSeek, Ollama, etc.)
  - local: In-process HuggingFace model managed by ModelManager (loads/unloads on demand)

Configuration via environment variables:
  LLM_PROVIDER=dashscope|openai|local
  DASHSCOPE_API_KEY=...
  OPENAI_API_KEY=...
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_MODEL=gpt-4o
  LOCAL_LLM_HF_ID=Qwen/Qwen3-8B-Instruct
  LOCAL_LLM_QUANT=auto|fp16|bf16|8bit|4bit
  LOCAL_LLM_IDLE_SEC=3
"""
```

- [ ] **Step 2: Verify the file still imports cleanly**

```bash
.venv/Scripts/python.exe -c "from src.apps.comic_gen.llm_adapter import LLMAdapter; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/apps/comic_gen/llm_adapter.py
git commit -m "feat(llm_adapter): add local provider branch dispatching to ModelManager"
```

---

## Task 8: Wire router into main api.py and install ModelManager at startup

**Files:**
- Modify: `src/apps/comic_gen/api.py`

- [ ] **Step 1: Import the local LLM module early**

The package import sets `HF_HOME`. We also need to install the singleton manager and capture the event loop.

Edit [src/apps/comic_gen/api.py](../../src/apps/comic_gen/api.py). After the existing imports block (after line 28, after `from dotenv import load_dotenv, set_key`), add:

```python
# Local LLM runtime — package import sets HF_HOME to <project>/output/models/LLM/
import src.llm_local  # noqa: F401  (side-effect import for HF_HOME)
from src.llm_local.api import router as local_llm_router
from src.llm_local.config import LocalLLMConfig
from src.llm_local.manager import ModelManager
```

- [ ] **Step 2: Install the manager singleton and include the router**

Locate the line `app = FastAPI(title="AI Comic Gen API")` (currently around line 30) and add immediately after the existing `app.add_middleware(CORSMiddleware, ...)` block (around line 47–54):

```python
# Install singleton ModelManager and mount /llm/local/* router.
ModelManager.install(LocalLLMConfig.from_env())
app.include_router(local_llm_router)


@app.on_event("startup")
async def _local_llm_startup():
    """Capture event loop for sync→async bridge and start idle watcher."""
    mgr = ModelManager.get()
    mgr.capture_loop()
    await mgr.start_idle_watcher()


@app.on_event("shutdown")
async def _local_llm_shutdown():
    mgr = ModelManager.get()
    await mgr.stop_idle_watcher()
    await mgr.unload()
```

- [ ] **Step 3: Verify the backend still boots**

The dev server has `--reload`, so saving the file should trigger restart. Watch the dev log:

```bash
tail -30 /c/Users/hemin/AppData/Local/Temp/claude/s--AI-lumenxDev/89649464-fa7b-4d17-a0d3-36c10b8729d6/tasks/bwbslzs0p.output
```

Expected to see:
- `INFO:     Started server process`
- `INFO:     Application startup complete`
- No traceback

If the dev server isn't running, start it: `cd /s/AI/lumenxDev/lumenx && npm run dev` (background).

- [ ] **Step 4: Hit /llm/local/status to confirm the route is mounted**

```bash
curl -s http://localhost:17177/llm/local/status | python -m json.tool
```

Expected output:
```json
{
    "state": "UNLOADED",
    "hf_id": "",
    "quant_mode": "auto",
    "vram_used_mb": 0,
    "vram_total_mb": 24576,
    "last_used_ts": <some_float>,
    "idle_seconds": 3,
    "error": null
}
```

(`vram_total_mb` will be the actual GPU memory in MB.)

- [ ] **Step 5: Commit**

```bash
git add src/apps/comic_gen/api.py
git commit -m "feat(api): mount local LLM router and install ModelManager singleton"
```

---

## Task 9: Strip response_format for local provider in llm.py

**Files:**
- Modify: `src/apps/comic_gen/llm.py`

The existing chat calls in `ScriptProcessor` don't currently pass `response_format`, but we should make the adapter behave correctly when callers do. The current adapter already silently ignores `response_format` for local (Task 7). However, some upstream callers may also depend on JSON parsing. The existing `_strip_markdown_json` helper already handles markdown-wrapped JSON, so no change is required to `llm.py` itself — but we should add a docstring note.

- [ ] **Step 1: Add a comment near _strip_markdown_json**

Edit [src/apps/comic_gen/llm.py](../../src/apps/comic_gen/llm.py) lines 13–19. Replace the existing function with:

```python
def _strip_markdown_json(content: str) -> str:
    """Strip markdown code fences from LLM JSON output.

    Local provider relies on this — it has no native JSON mode and may wrap
    responses in ```json fences depending on the model's training.
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()
```

- [ ] **Step 2: Commit**

```bash
git add src/apps/comic_gen/llm.py
git commit -m "docs(llm): note that _strip_markdown_json supports local provider output"
```

---

## Task 10: Frontend API client extensions

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add types and client functions**

Edit [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts). Append the following (after the existing `export const api = { ... }` block and before any default export, or merge into the existing `api` object — match the existing style):

First, add type exports near the top (after the `EnvConfigPayload` interface, around line 45):

```typescript
export type LocalLLMState =
    | "UNLOADED"
    | "DOWNLOADING"
    | "LOADING"
    | "READY"
    | "ERROR";

export type LocalLLMQuant = "auto" | "fp16" | "bf16" | "8bit" | "4bit";

export interface LocalLLMStatus {
    state: LocalLLMState;
    hf_id: string;
    quant_mode: string;
    vram_used_mb: number;
    vram_total_mb: number;
    last_used_ts: number;
    idle_seconds: number;
    error: string | null;
}

export interface LocalLLMConfig {
    hf_id: string;
    quant: LocalLLMQuant;
    idle_seconds: number;
}

export interface CachedModelInfo {
    hf_id: string;
    size_bytes: number;
    snapshot_path: string;
}
```

Then add API client functions. Inside the existing `export const api = {` object, add these methods (use the existing axios pattern — follow the surrounding style for `getEnvConfig` / `saveEnvConfig`):

```typescript
    // ── Local LLM ──────────────────────────────────────────
    getLocalLLMStatus: async (): Promise<LocalLLMStatus> => {
        const res = await axios.get<LocalLLMStatus>(`${API_URL}/llm/local/status`);
        return res.data;
    },

    configureLocalLLM: async (cfg: LocalLLMConfig): Promise<{ ok: boolean; persisted: boolean }> => {
        const res = await axios.post(`${API_URL}/llm/local/configure`, cfg);
        return res.data;
    },

    loadLocalLLM: async (): Promise<LocalLLMStatus> => {
        const res = await axios.post<LocalLLMStatus>(`${API_URL}/llm/local/load`, {}, { timeout: 600_000 });
        return res.data;
    },

    unloadLocalLLM: async (): Promise<LocalLLMStatus> => {
        const res = await axios.post<LocalLLMStatus>(`${API_URL}/llm/local/unload`);
        return res.data;
    },

    testLocalLLM: async (): Promise<{ ok: boolean; elapsed_sec: number; response: string }> => {
        const res = await axios.post(`${API_URL}/llm/local/test`, {}, { timeout: 600_000 });
        return res.data;
    },

    listCachedLLMs: async (): Promise<CachedModelInfo[]> => {
        const res = await axios.get<CachedModelInfo[]>(`${API_URL}/llm/local/cached`);
        return res.data;
    },

    deleteCachedLLM: async (hf_id: string): Promise<{ ok: boolean; deleted: string }> => {
        const res = await axios.delete(`${API_URL}/llm/local/cached`, { params: { hf_id } });
        return res.data;
    },
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): add Local LLM API client functions and types"
```

---

## Task 11: LocalLLMPanel React component

**Files:**
- Create: `frontend/src/components/settings/LocalLLMPanel.tsx`

- [ ] **Step 1: Create the panel component**

Create [frontend/src/components/settings/LocalLLMPanel.tsx](../../frontend/src/components/settings/LocalLLMPanel.tsx):

```tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import {
    api,
    type LocalLLMStatus,
    type LocalLLMQuant,
    type CachedModelInfo,
} from "@/lib/api";

const stateBadgeColors: Record<string, string> = {
    UNLOADED: "bg-gray-600",
    DOWNLOADING: "bg-blue-600 animate-pulse",
    LOADING: "bg-blue-600 animate-pulse",
    READY: "bg-green-600",
    ERROR: "bg-red-600",
};

function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function formatRelativeTime(ts: number): string {
    if (!ts) return "never";
    const diff = Date.now() / 1000 - ts;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
}

export default function LocalLLMPanel() {
    const [hfId, setHfId] = useState("");
    const [quant, setQuant] = useState<LocalLLMQuant>("auto");
    const [idleSec, setIdleSec] = useState(3);
    const [status, setStatus] = useState<LocalLLMStatus | null>(null);
    const [cached, setCached] = useState<CachedModelInfo[]>([]);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState<string | null>(null);

    const refreshStatus = useCallback(async () => {
        try {
            const s = await api.getLocalLLMStatus();
            setStatus(s);
            // Hydrate form from server on first load
            if (s.hf_id && !hfId) setHfId(s.hf_id);
            if (s.idle_seconds && idleSec === 3) setIdleSec(s.idle_seconds);
        } catch (e) {
            console.error("status fetch failed", e);
        }
    }, [hfId, idleSec]);

    const refreshCached = useCallback(async () => {
        try {
            setCached(await api.listCachedLLMs());
        } catch (e) {
            console.error("cached fetch failed", e);
        }
    }, []);

    useEffect(() => {
        refreshStatus();
        refreshCached();
        const id = setInterval(refreshStatus, 2000);
        return () => clearInterval(id);
    }, [refreshStatus, refreshCached]);

    const onSave = async () => {
        setBusy(true);
        setMessage(null);
        try {
            await api.configureLocalLLM({ hf_id: hfId, quant, idle_seconds: idleSec });
            setMessage("✓ Configuration saved");
            await refreshStatus();
        } catch (e: any) {
            setMessage(`✗ ${e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onLoad = async () => {
        setBusy(true);
        setMessage(null);
        try {
            await api.loadLocalLLM();
            setMessage("✓ Model loaded");
            await refreshStatus();
        } catch (e: any) {
            setMessage(`✗ ${e.response?.data?.detail || e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onUnload = async () => {
        setBusy(true);
        try {
            await api.unloadLocalLLM();
            await refreshStatus();
        } finally {
            setBusy(false);
        }
    };

    const onTest = async () => {
        setBusy(true);
        setMessage(null);
        try {
            const res = await api.testLocalLLM();
            setMessage(`✓ Test passed in ${res.elapsed_sec}s — "${res.response.slice(0, 80)}"`);
            await refreshStatus();
        } catch (e: any) {
            setMessage(`✗ ${e.response?.data?.detail || e.message || e}`);
        } finally {
            setBusy(false);
        }
    };

    const onDeleteCached = async (id: string) => {
        if (!confirm(`Delete cached model ${id}? This frees disk space.`)) return;
        try {
            await api.deleteCachedLLM(id);
            await refreshCached();
        } catch (e: any) {
            alert(`Delete failed: ${e.response?.data?.detail || e.message || e}`);
        }
    };

    return (
        <div className="space-y-6 rounded-lg bg-gray-900 p-4">
            <h3 className="text-sm font-bold text-white">Local LLM (HuggingFace)</h3>

            {/* Configuration form */}
            <div className="space-y-3">
                <div>
                    <label className="block text-xs text-gray-400 mb-1">HF Model ID</label>
                    <input
                        type="text"
                        value={hfId}
                        onChange={(e) => setHfId(e.target.value)}
                        placeholder="Qwen/Qwen3-8B-Instruct"
                        className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                    />
                </div>
                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Quantization</label>
                        <select
                            value={quant}
                            onChange={(e) => setQuant(e.target.value as LocalLLMQuant)}
                            className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                        >
                            <option value="auto">Auto</option>
                            <option value="fp16">FP16</option>
                            <option value="bf16">BF16</option>
                            <option value="8bit">8-bit</option>
                            <option value="4bit">4-bit</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">Idle unload (s)</label>
                        <input
                            type="number"
                            min={1}
                            max={3600}
                            value={idleSec}
                            onChange={(e) => setIdleSec(parseInt(e.target.value, 10) || 3)}
                            className="w-full rounded bg-gray-800 px-3 py-2 text-sm text-white"
                        />
                    </div>
                </div>
                <button
                    type="button"
                    disabled={busy || !hfId}
                    onClick={onSave}
                    className="rounded bg-blue-600 px-4 py-2 text-sm text-white disabled:opacity-50"
                >
                    Save Configuration
                </button>
            </div>

            <hr className="border-gray-700" />

            {/* Status block */}
            {status && (
                <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex items-center gap-2">
                        <span className={`inline-block h-2 w-2 rounded-full ${stateBadgeColors[status.state] ?? "bg-gray-600"}`} />
                        <span className="font-mono text-white">{status.state}</span>
                        {status.error && <span className="text-red-400 text-xs">— {status.error}</span>}
                    </div>
                    <div>Model: <span className="font-mono">{status.hf_id || "—"}</span></div>
                    <div>
                        VRAM: {(status.vram_used_mb / 1024).toFixed(1)} / {(status.vram_total_mb / 1024).toFixed(1)} GB
                    </div>
                    <div>Last used: {formatRelativeTime(status.last_used_ts)}</div>
                </div>
            )}

            <div className="flex gap-2">
                <button type="button" onClick={onLoad} disabled={busy || !hfId} className="rounded bg-green-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Load Now
                </button>
                <button type="button" onClick={onUnload} disabled={busy} className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Unload
                </button>
                <button type="button" onClick={onTest} disabled={busy || !hfId} className="rounded bg-purple-700 px-3 py-1.5 text-xs text-white disabled:opacity-50">
                    Test
                </button>
            </div>

            {message && (
                <div className="rounded bg-gray-800 px-3 py-2 text-xs text-gray-200">{message}</div>
            )}

            <hr className="border-gray-700" />

            {/* Cached models */}
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <h4 className="text-xs font-bold text-white">Cached Models (output/models/LLM/)</h4>
                    <button onClick={refreshCached} className="text-xs text-blue-400 hover:underline">Refresh</button>
                </div>
                {cached.length === 0 ? (
                    <div className="text-xs text-gray-500 italic">No models downloaded yet.</div>
                ) : (
                    <ul className="space-y-1">
                        {cached.map((m) => (
                            <li key={m.hf_id} className="flex items-center justify-between rounded bg-gray-800 px-3 py-1.5 text-xs">
                                <span className="font-mono text-gray-200">{m.hf_id}</span>
                                <div className="flex items-center gap-3">
                                    <span className="text-gray-400">{formatBytes(m.size_bytes)}</span>
                                    <button
                                        onClick={() => onDeleteCached(m.hf_id)}
                                        className="text-red-400 hover:text-red-300"
                                        title="Delete cached model"
                                    >
                                        🗑
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/settings/LocalLLMPanel.tsx
git commit -m "feat(frontend): add LocalLLMPanel component for HF model management UI"
```

---

## Task 12: Embed LocalLLMPanel into SettingsPage with provider selector

**Files:**
- Modify: `frontend/src/components/settings/SettingsPage.tsx`

- [ ] **Step 1: Add LLM Provider section to SettingsPage**

Edit [frontend/src/components/settings/SettingsPage.tsx](../../frontend/src/components/settings/SettingsPage.tsx).

Add the import at the top of the file (with the other imports):

```tsx
import LocalLLMPanel from "./LocalLLMPanel";
```

Add a state hook for the LLM provider near the other useState calls inside the component:

```tsx
const [llmProvider, setLlmProvider] = useState<"dashscope" | "openai" | "local">("dashscope");
```

In the existing `useEffect` that hydrates env config (look for `getEnvConfig` calls), add hydration of `llmProvider`:

```tsx
// Inside the existing hydration logic:
if (envData.LLM_PROVIDER === "openai" || envData.LLM_PROVIDER === "local") {
    setLlmProvider(envData.LLM_PROVIDER);
} else {
    setLlmProvider("dashscope");
}
```

Then add a new section in the JSX. Locate the existing "Kling Provider" section (around line 266) and add a new section ABOVE it:

```tsx
            <div className="rounded-lg bg-gray-900 p-4">
                <h3 className="text-sm font-bold text-white mb-4">LLM Provider (text generation)</h3>
                <div className="flex gap-2 mb-4">
                    <button
                        type="button"
                        onClick={() => setLlmProvider("dashscope")}
                        className={modeButtonClass(llmProvider === "dashscope")}
                    >
                        DashScope
                    </button>
                    <button
                        type="button"
                        onClick={() => setLlmProvider("openai")}
                        className={modeButtonClass(llmProvider === "openai")}
                    >
                        OpenAI-Compatible
                    </button>
                    <button
                        type="button"
                        onClick={() => setLlmProvider("local")}
                        className={modeButtonClass(llmProvider === "local")}
                    >
                        Local (HuggingFace)
                    </button>
                </div>
                {llmProvider === "local" && <LocalLLMPanel />}
                {llmProvider === "openai" && (
                    <p className="text-xs text-gray-500">
                        Configure OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL in .env, then save and restart.
                    </p>
                )}
                {llmProvider === "dashscope" && (
                    <p className="text-xs text-gray-500">
                        Uses DASHSCOPE_API_KEY (default). Configure your key in the API Keys section above.
                    </p>
                )}
            </div>
```

(`modeButtonClass` is already defined in this file — reuse it.)

When the user selects DashScope or OpenAI and clicks the global Save button (which already exists), persist `LLM_PROVIDER`. Find the existing save handler (look for `saveEnvConfig`) and ensure it includes `LLM_PROVIDER: llmProvider` in the payload. If `LocalLLMPanel` is shown, the user will save that separately via its own button — that's fine.

In the existing save handler, add:

```tsx
const payload = {
    // ... existing fields ...
    LLM_PROVIDER: llmProvider,
};
```

- [ ] **Step 2: Verify the frontend renders**

Open http://localhost:3000 and navigate to Settings. You should see a new "LLM Provider (text generation)" section with three buttons. Clicking "Local (HuggingFace)" should reveal the `LocalLLMPanel` component.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/settings/SettingsPage.tsx
git commit -m "feat(frontend): add LLM Provider switcher with Local panel embedded"
```

---

## Task 13: Update .env.example documentation

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add the LOCAL_LLM_* keys to the example**

Edit [.env.example](../../.env.example). Append after the existing `# LLM 多模型适配 (可选)` block (around line 40, after the Ollama example):

```
#
# 本地 HF 模型示例（应用内置加载/释放）：
# LLM_PROVIDER=local
# LOCAL_LLM_HF_ID=Qwen/Qwen3-8B-Instruct
# LOCAL_LLM_QUANT=auto                 # auto | fp16 | bf16 | 8bit | 4bit
# LOCAL_LLM_IDLE_SEC=3                 # seconds of idle before VRAM is freed
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs(.env.example): document LOCAL_LLM_* variables"
```

---

## Task 14: End-to-end smoke test with a real tiny model

**Files:**
- (No file changes; manual verification)

- [ ] **Step 1: Configure a tiny model via the UI**

Open http://localhost:3000/settings. Pick **Local (HuggingFace)**. Enter `Qwen/Qwen2.5-0.5B-Instruct`, leave quant on **Auto**, leave idle on **3**. Click **Save Configuration**.

Expected: `✓ Configuration saved`. Status badge shows UNLOADED.

- [ ] **Step 2: Click Load Now**

Expected: badge transitions UNLOADED → LOADING → READY. First time downloads ~1 GB; subsequent loads use the cache and complete in ~5 seconds. VRAM usage updates.

- [ ] **Step 3: Click Test**

Expected: `✓ Test passed in <1s — "<some greeting>"`.

- [ ] **Step 4: Wait 5 seconds, observe auto-unload**

Without clicking anything, status should transition to UNLOADED within 4–5 seconds (3s idle + ~1s watcher tick).

- [ ] **Step 5: Verify cached models list shows the model**

Refresh the cached list. Should display:
```
Qwen/Qwen2.5-0.5B-Instruct      ~1.0 GB    [🗑]
```

Verify the file is on disk:

```bash
ls /s/AI/lumenxDev/lumenx/output/models/LLM/hub/
```

Expected: a directory `models--Qwen--Qwen2.5-0.5B-Instruct/`.

- [ ] **Step 6: Run the full pipeline using the local model**

In the LumenX UI, create a new project, paste a short script, and click "Extract entities" (or whichever LLM-using button is available). The model should auto-load on first call.

Expected: extraction succeeds. Backend log shows:
```
INFO   Loading Qwen/Qwen2.5-0.5B-Instruct (quant=bf16)
INFO   Loaded Qwen/Qwen2.5-0.5B-Instruct in <N>s
```

(0.5B is too small for high-quality extraction — it's a smoke test only. Real use will require 8B+.)

- [ ] **Step 7: Test the delete cached model button**

In the cached list, click 🗑 next to the test model. Confirm the prompt. The model directory should be removed.

```bash
ls /s/AI/lumenxDev/lumenx/output/models/LLM/hub/
```

Expected: directory no longer present.

- [ ] **Step 8: Commit any incidental fixes found during smoke test**

If the smoke test surfaced any small issues (typos, off-by-one bugs, missing error handling for edge cases observed only at runtime), fix them and commit:

```bash
git add <files>
git commit -m "fix(llm_local): <specific fix>"
```

---

## Task 15: Run the full test suite

- [ ] **Step 1: Run all llm_local tests**

```bash
cd /s/AI/lumenxDev/lumenx && .venv/Scripts/python.exe -m pytest tests/llm_local/ -v
```

Expected: all tests pass (~22 tests). Skipped: 4 if no GPU.

- [ ] **Step 2: Run the existing test_pipeline (sanity check we didn't break it)**

```bash
.venv/Scripts/python.exe -m pytest src/apps/comic_gen/test_pipeline.py -v
```

Expected: same pass rate as before this work (or skip with reason if it requires a real DashScope key).

- [ ] **Step 3: Final commit if any test fixes were needed**

```bash
git add tests/
git commit -m "test: stabilize llm_local test suite"
```

---

## Done

The local LLM runtime is now operational. Users can:
- Switch the LLM provider to **Local** in Settings
- Enter any HF instruct model id, save, and use it for all script-processing LLM tasks
- Watch VRAM auto-free after 3 seconds of idle
- Browse and delete cached models from the project's `output/models/LLM/` directory

Next-phase candidates (out of scope for this plan, but the architecture supports them):
- **Image model manager**: same pattern, `output/models/Image/`, e.g. Flux / SDXL via diffusers
- **Video model manager**: same pattern, `output/models/Video/`, e.g. Wan2.1 via diffusers
- **Per-task model selection**: route entity-extraction → small model, prompt-polish → large model
- **HF download progress**: stream bytes-downloaded into `/llm/local/status` for UI progress bar
- **Streaming chat**: extend `LLMAdapter.chat()` with `stream=True` variant
