# LLM Model Picker (in-modal) + GGUF Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-modal LLM picker section to `ModelSettingsModal` (cloud DashScope cards + dynamic local cards + "+ Add Local Model" with GGUF auto-download) and a new `GGUFRuntime` (llama-cpp-python) that ModelManager dispatches to for any GGUF HF repo.

**Architecture:** New `src/llm_local/{gguf_utils,runtime_gguf}.py`. ModelManager's runtime factory now branches on `is_gguf_repo(hf_id)` → GGUFRuntime else existing LocalRuntime. LLMAdapter's DashScope branch reads a new `DASHSCOPE_MODEL` env var. Frontend gets `LLMModelSection.tsx` embedded in the existing `ModelSettingsModal`, fetching state from `/config/env` + `/llm/local/cached`.

**Tech Stack:** Python 3.12, FastAPI, asyncio, llama-cpp-python (CUDA), huggingface_hub, transformers (existing), Next.js 14, React 18, TypeScript, framer-motion (existing).

**Reference spec:** [docs/superpowers/specs/2026-04-25-llm-picker-and-gguf-runtime-design.md](../specs/2026-04-25-llm-picker-and-gguf-runtime-design.md)

---

## Setup Notes

- All Python paths assume project root `s:\AI\lumenxDev\lumenx`. Always `cd` first when invoking `.venv/Scripts/python.exe`.
- Dev server is started by `npm run dev` (backend reload-aware, frontend HMR). Restart only if a Python module's top-level imports change.
- `bash` is the default shell; use Unix path syntax (`/dev/null`).
- New tests live under `tests/llm_local/`. Run all with `.venv/Scripts/python.exe -m pytest tests/llm_local/ -v`.
- Commits use Conventional Commits style (`feat:`, `chore:`, `test:`, `docs:`, `fix:`).

---

## File Structure

**New files:**

```
src/llm_local/
├── gguf_utils.py            # is_gguf_repo, list_gguf_files, pick_default_gguf_file
└── runtime_gguf.py          # GGUFRuntime: llama_cpp wrapper

tests/llm_local/
├── test_gguf_utils.py       # Pure logic tests
├── test_runtime_gguf.py     # Optional integration test (small GGUF, GPU-only)
└── test_llm_adapter.py      # NEW: DASHSCOPE_MODEL env var support

frontend/src/components/common/
└── LLMModelSection.tsx      # The new LLM picker section
```

**Modified files:**

```
src/llm_local/config.py                                # add gguf_file: Optional[str]
src/llm_local/manager.py                               # factory dispatch on GGUF; pass gguf_file
src/llm_local/api.py                                   # configure accepts gguf_file; cached returns gguf_files+active_gguf_file
src/apps/comic_gen/llm_adapter.py                      # _get_default_model() reads DASHSCOPE_MODEL
requirements.txt                                       # add llama-cpp-python
.env.example                                           # document DASHSCOPE_MODEL + LOCAL_LLM_GGUF_FILE
frontend/src/lib/api.ts                                # extend LocalLLMConfig with gguf_file?, CachedModelInfo with gguf_files+active_gguf_file
frontend/src/components/common/ModelSettingsModal.tsx  # embed <LLMModelSection />
tests/llm_local/test_config.py                         # extend: gguf_file persistence
tests/llm_local/test_manager.py                        # extend: factory GGUF dispatch
tests/llm_local/test_api.py                            # extend: gguf_file roundtrip + cached enrichment
```

---

## Task 1: Install llama-cpp-python (CUDA wheel)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add llama-cpp-python to requirements.txt**

Edit [requirements.txt](../../requirements.txt). In the existing `# Local LLM Runtime` block, add one line so the block reads:

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
llama-cpp-python>=0.3.0
```

- [ ] **Step 2: Install the CUDA-enabled wheel**

Run from `s:\AI\lumenxDev\lumenx`:

```bash
.venv/Scripts/python.exe -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Expected last line: `Successfully installed llama-cpp-python-0.3.x` (some `0.3.*` version).

If the CUDA index has no wheel for Python 3.12 + cu121, fall back to PyPI (CPU-only build, slower but functional):

```bash
.venv/Scripts/python.exe -m pip install llama-cpp-python
```

- [ ] **Step 3: Verify import**

```bash
.venv/Scripts/python.exe -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

Expected: `llama-cpp-python OK` (no traceback).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add llama-cpp-python for GGUF runtime"
```

---

## Task 2: GGUF utilities (detection + file picking)

**Files:**
- Create: `src/llm_local/gguf_utils.py`
- Create: `tests/llm_local/test_gguf_utils.py`

- [ ] **Step 1: Write the failing test**

Create [tests/llm_local/test_gguf_utils.py](../../tests/llm_local/test_gguf_utils.py):

```python
"""Tests for GGUF detection and file priority."""
from unittest.mock import patch

from src.llm_local.gguf_utils import (
    is_gguf_repo,
    pick_default_gguf_file,
    list_gguf_files,
)


class TestIsGgufRepo:
    def test_recognises_gguf_suffix(self):
        assert is_gguf_repo("anthfu/Qwen3.6-35B-A3B-APEX-GGUF") is True

    def test_recognises_lowercase_gguf(self):
        assert is_gguf_repo("bartowski/Qwen2.5-0.5B-Instruct-gguf") is True

    def test_recognises_gguf_anywhere(self):
        assert is_gguf_repo("user/SomeModel-GGUF-Quantized") is True

    def test_rejects_non_gguf(self):
        assert is_gguf_repo("Qwen/Qwen3-8B-Instruct") is False
        assert is_gguf_repo("meta-llama/Llama-3-70B-Instruct") is False


class TestPickDefaultGgufFile:
    def test_picks_q4_k_m_first(self):
        files = ["model-Q5_K_M.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        assert pick_default_gguf_file(files) == "model-Q4_K_M.gguf"

    def test_picks_q5_k_m_when_no_q4_k_m(self):
        files = ["model-Q5_K_M.gguf", "model-Q8_0.gguf"]
        assert pick_default_gguf_file(files) == "model-Q5_K_M.gguf"

    def test_falls_back_to_first_when_no_priority_match(self):
        files = ["model-IQ3_XS.gguf", "model-IQ2_M.gguf"]
        assert pick_default_gguf_file(files) == "model-IQ3_XS.gguf"

    def test_returns_none_for_empty(self):
        assert pick_default_gguf_file([]) is None


class TestListGgufFiles:
    def test_filters_to_gguf_only(self):
        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            mock_api.return_value.list_repo_files.return_value = [
                ".gitattributes",
                "README.md",
                "model-Q4_K_M.gguf",
                "model-Q5_K_M.gguf",
                "config.json",
            ]
            assert list_gguf_files("anthfu/Qwen3.6-35B-A3B-APEX-GGUF") == [
                "model-Q4_K_M.gguf",
                "model-Q5_K_M.gguf",
            ]

    def test_returns_empty_when_no_gguf(self):
        with patch("src.llm_local.gguf_utils._hf_api") as mock_api:
            mock_api.return_value.list_repo_files.return_value = ["config.json", "model.safetensors"]
            assert list_gguf_files("Qwen/Qwen3-8B-Instruct") == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /s/AI/lumenxDev/lumenx && .venv/Scripts/python.exe -m pytest tests/llm_local/test_gguf_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.llm_local.gguf_utils'`.

- [ ] **Step 3: Implement the module**

Create [src/llm_local/gguf_utils.py](../../src/llm_local/gguf_utils.py):

```python
"""GGUF repo detection + .gguf file priority for auto-pick."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional


# Quant priority for the auto-picker — best balance for typical 24GB VRAM:
# Q4_K_M is the de facto "default" quant; the rest are fallbacks if it's missing.
_QUANT_PRIORITY = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0"]


@lru_cache(maxsize=1)
def _hf_api():
    """Lazy-construct an HfApi instance (avoids import at module load)."""
    from huggingface_hub import HfApi
    return HfApi()


def is_gguf_repo(hf_id: str) -> bool:
    """Cheap heuristic: hf_id contains 'gguf' (case-insensitive)."""
    return "gguf" in hf_id.lower()


def list_gguf_files(hf_id: str) -> List[str]:
    """Network call: list .gguf files in repo via huggingface_hub.HfApi."""
    files = _hf_api().list_repo_files(hf_id)
    return [f for f in files if f.endswith(".gguf")]


def pick_default_gguf_file(files: List[str]) -> Optional[str]:
    """Pick the highest-priority quant file, or the first .gguf file as fallback."""
    for quant in _QUANT_PRIORITY:
        for f in files:
            if quant in f:
                return f
    return files[0] if files else None
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_gguf_utils.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/gguf_utils.py tests/llm_local/test_gguf_utils.py
git commit -m "feat(llm_local): add GGUF detection and auto file picker"
```

---

## Task 3: Extend LocalLLMConfig with gguf_file

**Files:**
- Modify: `src/llm_local/config.py`
- Modify: `tests/llm_local/test_config.py`

- [ ] **Step 1: Extend the test file**

Edit [tests/llm_local/test_config.py](../../tests/llm_local/test_config.py). Add these tests at the end of `TestLocalLLMConfig`:

```python
    def test_defaults_include_no_gguf_file(self):
        cfg = LocalLLMConfig()
        assert cfg.gguf_file is None

    def test_from_env_reads_gguf_file(self):
        env = {"LOCAL_LLM_GGUF_FILE": "model-Q4_K_M.gguf"}
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.gguf_file == "model-Q4_K_M.gguf"

    def test_from_env_treats_empty_gguf_file_as_none(self):
        env = {"LOCAL_LLM_GGUF_FILE": ""}
        with patch.dict("os.environ", env, clear=False):
            cfg = LocalLLMConfig.from_env()
        assert cfg.gguf_file is None

    def test_to_env_dict_emits_gguf_file(self):
        cfg = LocalLLMConfig(
            hf_id="anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
            quant=QuantMode.AUTO,
            idle_seconds=3,
            gguf_file="model-Q4_K_M.gguf",
        )
        d = cfg.to_env_dict()
        assert d["LOCAL_LLM_GGUF_FILE"] == "model-Q4_K_M.gguf"

    def test_to_env_dict_emits_empty_when_no_gguf_file(self):
        cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct")
        d = cfg.to_env_dict()
        assert d["LOCAL_LLM_GGUF_FILE"] == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_config.py -v
```

Expected: 5 new tests fail with `AttributeError: ... 'gguf_file'` or `KeyError: 'LOCAL_LLM_GGUF_FILE'`.

- [ ] **Step 3: Update LocalLLMConfig**

Edit [src/llm_local/config.py](../../src/llm_local/config.py). Replace the entire file with:

```python
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

    @classmethod
    def from_env(cls) -> "LocalLLMConfig":
        """Hydrate from process environment (loaded from .env at startup)."""
        gguf_file = os.environ.get("LOCAL_LLM_GGUF_FILE", "") or None
        return cls(
            hf_id=os.environ.get("LOCAL_LLM_HF_ID", ""),
            quant=QuantMode(os.environ.get("LOCAL_LLM_QUANT", QuantMode.AUTO.value)),
            idle_seconds=int(os.environ.get("LOCAL_LLM_IDLE_SEC", "3")),
            gguf_file=gguf_file,
        )

    def to_env_dict(self) -> Dict[str, str]:
        """Render to env-style dict for persistence via save_user_config()."""
        return {
            "LOCAL_LLM_HF_ID": self.hf_id,
            "LOCAL_LLM_QUANT": self.quant.value,
            "LOCAL_LLM_IDLE_SEC": str(self.idle_seconds),
            "LOCAL_LLM_GGUF_FILE": self.gguf_file or "",
        }
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_config.py -v
```

Expected: 9 passed (4 original + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/config.py tests/llm_local/test_config.py
git commit -m "feat(llm_local): add gguf_file field to LocalLLMConfig"
```

---

## Task 4: GGUFRuntime (llama_cpp wrapper)

**Files:**
- Create: `src/llm_local/runtime_gguf.py`
- Create: `tests/llm_local/test_runtime_gguf.py`

- [ ] **Step 1: Write the integration test (skipped without GPU)**

Create [tests/llm_local/test_runtime_gguf.py](../../tests/llm_local/test_runtime_gguf.py):

```python
"""Integration tests for GGUFRuntime using a tiny GGUF model.

Skipped when no CUDA GPU; downloads ~400 MB on first run.
"""
import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False
    torch = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="GGUFRuntime is built for CUDA; skipped without GPU",
)

if _HAS_CUDA:
    from src.llm_local.runtime_gguf import GGUFRuntime
    from src.llm_local.runtime import LoadResult

# Tiny instruct GGUF (~400 MB at Q4_K_M) widely available on HF.
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
TEST_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


@pytest.fixture(scope="module")
def runtime():
    rt = GGUFRuntime(hf_id=TEST_MODEL, gguf_file=TEST_FILE)
    yield rt
    rt.unload()


class TestGGUFRuntime:
    def test_load_returns_load_result(self, runtime):
        result = runtime.load()
        assert isinstance(result, LoadResult)
        assert result.hf_id == TEST_MODEL

    def test_chat_returns_non_empty_string(self, runtime):
        runtime.load()
        out = runtime.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)
        assert isinstance(out, str)
        assert len(out.strip()) > 0

    def test_unload_is_idempotent(self, runtime):
        runtime.load()
        runtime.unload()
        runtime.unload()  # second call must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_runtime_gguf.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.llm_local.runtime_gguf'`.

- [ ] **Step 3: Implement GGUFRuntime**

Create [src/llm_local/runtime_gguf.py](../../src/llm_local/runtime_gguf.py):

```python
"""GGUFRuntime — synchronous llama-cpp-python wrapper for one GGUF model at a time."""
from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Optional

from .runtime import LoadResult
from .vram import QuantMode

logger = logging.getLogger(__name__)


class GGUFRuntime:
    """Owns a single loaded GGUF model. Pure sync; ModelManager owns concurrency.

    Implements the same minimal interface as LocalRuntime:
      - load() -> LoadResult
      - chat(messages, max_new_tokens=, temperature=) -> str
      - unload() -> None
    """

    def __init__(self, hf_id: str, gguf_file: str, n_ctx: int = 8192):
        self.hf_id = hf_id
        self.gguf_file = gguf_file
        self.n_ctx = n_ctx
        self.quant = QuantMode.AUTO  # not meaningful for GGUF; kept for interface parity
        self._llama = None
        self._loaded = False

    def load(self) -> LoadResult:
        """Idempotent. Downloads via hf_hub_download (uses HF_HOME cache) and loads via llama_cpp."""
        if self._loaded:
            return self._current_load_result(elapsed_sec=0.0)

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        t0 = time.monotonic()
        logger.info(f"Loading GGUF {self.hf_id}/{self.gguf_file} (n_ctx={self.n_ctx})")

        # hf_hub_download honours HF_HOME (set by src/llm_local/__init__.py)
        local_path = hf_hub_download(repo_id=self.hf_id, filename=self.gguf_file)

        self._llama = Llama(
            model_path=local_path,
            n_gpu_layers=-1,        # offload all layers to GPU
            n_ctx=self.n_ctx,
            verbose=False,
            chat_format=None,       # auto-detect from GGUF metadata
        )
        self._loaded = True

        elapsed = time.monotonic() - t0
        logger.info(f"Loaded GGUF {self.hf_id} in {elapsed:.1f}s")
        return self._current_load_result(elapsed_sec=elapsed)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("GGUF runtime not loaded; call load() first.")

        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"]

    def unload(self) -> None:
        if not self._loaded and self._llama is None:
            return
        logger.info(f"Unloading GGUF {self.hf_id}")
        # llama_cpp.Llama frees its internal context on __del__
        del self._llama
        self._llama = None
        self._loaded = False
        gc.collect()

    def _current_load_result(self, elapsed_sec: float) -> LoadResult:
        # llama_cpp does not expose VRAM usage directly; report 0 (documented limitation).
        return LoadResult(
            hf_id=self.hf_id,
            quant_mode=self.quant,
            vram_used_mb=0,
            elapsed_sec=elapsed_sec,
        )
```

- [ ] **Step 4: Run the integration test**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_runtime_gguf.py -v -s
```

Expected: 3 passed (first run downloads ~400MB; subsequent ~5s). If no GPU, all skipped.

- [ ] **Step 5: Commit**

```bash
git add src/llm_local/runtime_gguf.py tests/llm_local/test_runtime_gguf.py
git commit -m "feat(llm_local): add GGUFRuntime backed by llama-cpp-python"
```

---

## Task 5: ModelManager — factory dispatch on GGUF

**Files:**
- Modify: `src/llm_local/manager.py`
- Modify: `tests/llm_local/test_manager.py`

- [ ] **Step 1: Add a failing test for GGUF dispatch**

Edit [tests/llm_local/test_manager.py](../../tests/llm_local/test_manager.py). Add at the bottom of the file:

```python
@pytest.mark.asyncio
async def test_factory_dispatches_gguf_repo_to_gguf_runtime(stub_load_result):
    """When hf_id is a GGUF repo, the factory must receive a gguf_file kwarg."""
    cfg = LocalLLMConfig(
        hf_id="anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
        quant=QuantMode.AUTO,
        idle_seconds=3,
        gguf_file="model-Q4_K_M.gguf",
    )
    received_kwargs = {}

    def factory(hf_id, quant, gguf_file=None):
        received_kwargs["hf_id"] = hf_id
        received_kwargs["gguf_file"] = gguf_file
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = stub_load_result
        rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    await mgr.chat([{"role": "user", "content": "hi"}])
    assert received_kwargs["hf_id"] == "anthfu/Qwen3.6-35B-A3B-APEX-GGUF"
    assert received_kwargs["gguf_file"] == "model-Q4_K_M.gguf"


@pytest.mark.asyncio
async def test_factory_does_not_pass_gguf_file_for_non_gguf(stub_load_result):
    """For non-GGUF repos, gguf_file kwarg is None."""
    cfg = LocalLLMConfig(hf_id="Qwen/Qwen3-8B-Instruct", quant=QuantMode.BF16, idle_seconds=3)
    received_kwargs = {}

    def factory(hf_id, quant, gguf_file=None):
        received_kwargs["gguf_file"] = gguf_file
        rt = MagicMock()
        rt.hf_id = hf_id
        rt.quant = quant
        rt.load.return_value = stub_load_result
        rt.chat.return_value = "OK"
        return rt

    mgr = ModelManager(config=cfg, runtime_factory=factory)
    await mgr.chat([{"role": "user", "content": "hi"}])
    assert received_kwargs["gguf_file"] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_manager.py -v -k "gguf"
```

Expected: 2 fail. The current factory signature only accepts `(hf_id, quant)` — calling with `gguf_file=` will raise TypeError, OR the manager doesn't pass it through.

- [ ] **Step 3: Update the default factory**

Edit [src/llm_local/manager.py](../../src/llm_local/manager.py). Replace the existing `_default_runtime_factory` block (which currently just returns `LocalRuntime(hf_id=hf_id, quant=quant)`) with:

```python
RuntimeFactory = Callable[..., "LocalRuntime"]


def _default_runtime_factory(hf_id: str, quant: QuantMode, gguf_file: Optional[str] = None):
    """Dispatch to GGUFRuntime for GGUF repos, else LocalRuntime (transformers)."""
    from .gguf_utils import is_gguf_repo, list_gguf_files, pick_default_gguf_file

    if is_gguf_repo(hf_id):
        from .runtime_gguf import GGUFRuntime
        if gguf_file is None:
            files = list_gguf_files(hf_id)
            gguf_file = pick_default_gguf_file(files)
            if gguf_file is None:
                raise RuntimeError(f"No .gguf files found in repo {hf_id}")
        return GGUFRuntime(hf_id=hf_id, gguf_file=gguf_file)

    return LocalRuntime(hf_id=hf_id, quant=quant)
```

(The `RuntimeFactory` type alias becomes `Callable[..., LocalRuntime]` — `...` means any args. This keeps existing test stubs valid.)

Add `Optional` to the existing typing imports near the top of the file (find `from typing import ...` and ensure `Optional` is there — it already should be).

- [ ] **Step 4: Update `_load_locked` to pass gguf_file**

Still in [src/llm_local/manager.py](../../src/llm_local/manager.py), find the line in `_load_locked` that calls the factory:

```python
self._runtime = self._runtime_factory(self.config.hf_id, attempt_quant)
```

Replace with:

```python
self._runtime = self._runtime_factory(
    self.config.hf_id,
    attempt_quant,
    gguf_file=self.config.gguf_file,
)
```

- [ ] **Step 5: Skip OOM step-down for GGUF**

GGUF quant is baked into the file — no step-down possible. In `_load_locked`, after the existing `_OOM_STEP_DOWN` branch, before `attempt_quant = next_quant`, add a guard:

Find this section:

```python
if self._is_oom_error(e):
    next_quant = self._OOM_STEP_DOWN.get(attempt_quant)
    if next_quant is not None:
        logger.warning(
            f"OOM at {attempt_quant.value}; stepping down to {next_quant.value}"
        )
        attempt_quant = next_quant
        continue
```

Replace with:

```python
if self._is_oom_error(e):
    from .gguf_utils import is_gguf_repo
    if is_gguf_repo(self.config.hf_id):
        logger.error(f"OOM loading GGUF {self.config.hf_id}; cannot step-down (quant baked into file)")
    else:
        next_quant = self._OOM_STEP_DOWN.get(attempt_quant)
        if next_quant is not None:
            logger.warning(
                f"OOM at {attempt_quant.value}; stepping down to {next_quant.value}"
            )
            attempt_quant = next_quant
            continue
```

- [ ] **Step 6: Update existing test stubs to accept gguf_file kwarg**

The existing test factories in `tests/llm_local/test_manager.py` are `def factory(hf_id, quant):` — they will TypeError when ModelManager passes the new `gguf_file=` kwarg. Update them now (before re-running tests).

In `make_stub_runtime_factory` (top of file), change:
```python
def factory(hf_id, quant):
```
to:
```python
def factory(hf_id, quant, gguf_file=None):
```

Also in `test_load_oom_steps_down_quant` and `test_load_non_oom_error_does_not_step_down`, change their inline `def factory(hf_id, quant):` to `def factory(hf_id, quant, gguf_file=None):`.

- [ ] **Step 7: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_manager.py -v
```

Expected: 11 passed (9 original + 2 new).

- [ ] **Step 8: Commit**

```bash
git add src/llm_local/manager.py tests/llm_local/test_manager.py
git commit -m "feat(llm_local): dispatch GGUF repos to GGUFRuntime in ModelManager factory"
```

---

## Task 6: API — accept gguf_file in configure, return it in cached

**Files:**
- Modify: `src/llm_local/api.py`
- Modify: `tests/llm_local/test_api.py`

- [ ] **Step 1: Add failing tests for the API extensions**

Edit [tests/llm_local/test_api.py](../../tests/llm_local/test_api.py). Add inside `class TestApi`:

```python
    def test_configure_accepts_gguf_file(self, client):
        payload = {
            "hf_id": "anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
            "quant": "auto",
            "idle_seconds": 3,
            "gguf_file": "model-Q4_K_M.gguf",
        }
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        # The persisted dict must include LOCAL_LLM_GGUF_FILE
        saved = mock_save.call_args[0][0]
        assert saved["LOCAL_LLM_GGUF_FILE"] == "model-Q4_K_M.gguf"

    def test_configure_omitted_gguf_file_persists_empty(self, client):
        payload = {"hf_id": "Qwen/Qwen3-8B-Instruct", "quant": "auto", "idle_seconds": 3}
        with patch("src.llm_local.api.save_user_config") as mock_save:
            resp = client.post("/llm/local/configure", json=payload)
        assert resp.status_code == 200
        saved = mock_save.call_args[0][0]
        assert saved["LOCAL_LLM_GGUF_FILE"] == ""

    def test_cached_lists_gguf_files_for_gguf_repo(self, client, tmp_path, monkeypatch):
        # Build a fake HF cache layout with a GGUF file
        cache = tmp_path / "models--anthfu--Qwen3.6-35B-A3B-APEX-GGUF" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "model-Q4_K_M.gguf").write_bytes(b"x" * 1000)
        (cache / "model-Q5_K_M.gguf").write_bytes(b"x" * 2000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)

        # Mock manager status so active_gguf_file is set
        from src.llm_local.manager import ModelManager
        mgr_mock = MagicMock(spec=ModelManager)
        mgr_mock.status.return_value = {
            "state": "READY",
            "hf_id": "anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
            "quant_mode": "auto",
            "vram_used_mb": 0,
            "vram_total_mb": 24576,
            "last_used_ts": 0,
            "idle_seconds": 3,
            "error": None,
        }
        # Manager's config must expose gguf_file
        mgr_mock.config.gguf_file = "model-Q4_K_M.gguf"
        with patch("src.llm_local.api.ModelManager.get", return_value=mgr_mock):
            resp = client.get("/llm/local/cached")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["hf_id"] == "anthfu/Qwen3.6-35B-A3B-APEX-GGUF"
        assert sorted(body[0]["gguf_files"]) == ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
        assert body[0]["active_gguf_file"] == "model-Q4_K_M.gguf"

    def test_cached_returns_null_gguf_for_non_gguf_repo(self, client, tmp_path, monkeypatch):
        cache = tmp_path / "models--Qwen--Qwen3-8B-Instruct" / "snapshots" / "abc"
        cache.mkdir(parents=True)
        (cache / "config.json").write_bytes(b"x" * 1000)
        monkeypatch.setattr("src.llm_local.api.HF_CACHE_DIR", tmp_path)
        resp = client.get("/llm/local/cached")
        assert resp.status_code == 200
        body = resp.json()
        assert body[0]["gguf_files"] is None
        assert body[0]["active_gguf_file"] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_api.py -v -k "gguf"
```

Expected: failures (the configure payload doesn't have gguf_file in the model; cached doesn't return gguf_files).

- [ ] **Step 3: Update ConfigurePayload and CachedModelInfo**

Edit [src/llm_local/api.py](../../src/llm_local/api.py). Replace the `ConfigurePayload` and `CachedModelInfo` classes with:

```python
class ConfigurePayload(BaseModel):
    hf_id: str = Field(..., min_length=1)
    quant: QuantMode = QuantMode.AUTO
    idle_seconds: int = Field(default=3, ge=1, le=3600)
    gguf_file: Optional[str] = Field(default=None)


class CachedModelInfo(BaseModel):
    hf_id: str
    size_bytes: int
    snapshot_path: str
    gguf_files: Optional[List[str]] = None
    active_gguf_file: Optional[str] = None
```

Make sure `Optional` is imported at the top of the file:

```python
from typing import List, Optional
```

- [ ] **Step 4: Update the configure handler**

Replace the existing `configure` handler with:

```python
@router.post("/configure")
async def configure(payload: ConfigurePayload) -> dict:
    cfg = LocalLLMConfig(
        hf_id=payload.hf_id,
        quant=payload.quant,
        idle_seconds=payload.idle_seconds,
        gguf_file=payload.gguf_file,
    )
    save_user_config({"LLM_PROVIDER": "local", **cfg.to_env_dict()})
    os.environ["LLM_PROVIDER"] = "local"
    for k, v in cfg.to_env_dict().items():
        os.environ[k] = v
    await ModelManager.get().configure(cfg)
    return {"ok": True, "persisted": True}
```

- [ ] **Step 5: Update list_cached to enrich GGUF metadata**

Replace the existing `list_cached` handler with:

```python
@router.get("/cached")
def list_cached() -> List[CachedModelInfo]:
    """Scan HF cache layout `models--<org>--<name>/snapshots/<rev>` and report sizes."""
    if not HF_CACHE_DIR.exists():
        return []

    from .gguf_utils import is_gguf_repo

    # Read active GGUF file from the running manager (best-effort; fall back to None)
    try:
        active_hf_id = ModelManager.get().status().get("hf_id", "")
        active_gguf = getattr(ModelManager.get().config, "gguf_file", None)
    except Exception:
        active_hf_id = ""
        active_gguf = None

    out: List[CachedModelInfo] = []
    for entry in HF_CACHE_DIR.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        parts = entry.name[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        hf_id = "/".join(parts)
        snapshots_dir = entry / "snapshots"
        if not snapshots_dir.exists():
            continue
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        size = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())

        # GGUF enrichment: list .gguf files actually present in the latest snapshot
        gguf_files: Optional[List[str]] = None
        active_for_this: Optional[str] = None
        if is_gguf_repo(hf_id):
            gguf_files = sorted(f.name for f in latest.iterdir() if f.is_file() and f.name.endswith(".gguf"))
            if hf_id == active_hf_id:
                active_for_this = active_gguf

        out.append(CachedModelInfo(
            hf_id=hf_id,
            size_bytes=size,
            snapshot_path=str(latest),
            gguf_files=gguf_files,
            active_gguf_file=active_for_this,
        ))
    return out
```

- [ ] **Step 6: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_api.py -v
```

Expected: 11 passed (7 original + 4 new).

- [ ] **Step 7: Commit**

```bash
git add src/llm_local/api.py tests/llm_local/test_api.py
git commit -m "feat(llm_local): wire gguf_file through configure + enrich /cached with GGUF metadata"
```

---

## Task 7: LLMAdapter — read DASHSCOPE_MODEL env var

**Files:**
- Modify: `src/apps/comic_gen/llm_adapter.py`
- Create: `tests/llm_local/test_llm_adapter.py`

- [ ] **Step 1: Write the failing test**

Create [tests/llm_local/test_llm_adapter.py](../../tests/llm_local/test_llm_adapter.py):

```python
"""Tests for LLMAdapter env var handling."""
import os
from unittest.mock import patch

from src.apps.comic_gen.llm_adapter import LLMAdapter


class TestDashscopeModel:
    def test_default_model_falls_back_when_env_unset(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "dashscope"}, clear=False):
            os.environ.pop("DASHSCOPE_MODEL", None)
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "qwen3.5-plus"

    def test_default_model_reads_dashscope_model_env(self):
        env = {"LLM_PROVIDER": "dashscope", "DASHSCOPE_MODEL": "qwen-max"}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "qwen-max"

    def test_openai_path_unchanged(self):
        env = {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4o"}
        with patch.dict(os.environ, env, clear=False):
            adapter = LLMAdapter()
            assert adapter._get_default_model() == "gpt-4o"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_llm_adapter.py -v
```

Expected: `test_default_model_reads_dashscope_model_env` fails — adapter returns the hardcoded `"qwen3.5-plus"` instead of reading the env var.

- [ ] **Step 3: Patch _get_default_model**

Edit [src/apps/comic_gen/llm_adapter.py](../../src/apps/comic_gen/llm_adapter.py). Replace `_get_default_model` (currently the single-line return for non-openai) with:

```python
    def _get_default_model(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_MODEL", "gpt-4o")
        return os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/llm_local/test_llm_adapter.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/apps/comic_gen/llm_adapter.py tests/llm_local/test_llm_adapter.py
git commit -m "feat(llm_adapter): support DASHSCOPE_MODEL env var override"
```

---

## Task 8: Update .env.example documentation

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Document new env vars**

Edit [.env.example](../../.env.example). Locate the existing local HF block:

```
# 本地 HF 模型示例（应用内置加载/释放，模型缓存在 output/models/LLM/）：
# LLM_PROVIDER=local
# LOCAL_LLM_HF_ID=Qwen/Qwen3-8B-Instruct
# LOCAL_LLM_QUANT=auto                 # auto | fp16 | bf16 | 8bit | 4bit
# LOCAL_LLM_IDLE_SEC=3                 # seconds of idle before VRAM is freed
```

Replace it with:

```
# 本地 HF 模型示例（应用内置加载/释放，模型缓存在 output/models/LLM/）：
# LLM_PROVIDER=local
# LOCAL_LLM_HF_ID=Qwen/Qwen3-8B-Instruct
# LOCAL_LLM_QUANT=auto                 # auto | fp16 | bf16 | 8bit | 4bit (transformers path only)
# LOCAL_LLM_IDLE_SEC=3                 # seconds of idle before VRAM is freed
# LOCAL_LLM_GGUF_FILE=                 # optional: specific .gguf file in a GGUF repo (auto-picked if empty)
#
# 本地 GGUF 示例（自动用 llama-cpp-python 加载）：
# LLM_PROVIDER=local
# LOCAL_LLM_HF_ID=anthfu/Qwen3.6-35B-A3B-APEX-GGUF
# LOCAL_LLM_GGUF_FILE=                 # 留空则自动按 Q4_K_M > Q5_K_M > ... 优先级挑

# DashScope 模型选择 (可选，默认 qwen3.5-plus):
# DASHSCOPE_MODEL=qwen-plus            # qwen3-max | qwen-plus | qwen-max | qwen-turbo 等
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs(.env.example): document DASHSCOPE_MODEL and LOCAL_LLM_GGUF_FILE"
```

---

## Task 9: Frontend — extend api.ts types

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Extend LocalLLMConfig with gguf_file**

Edit [frontend/src/lib/api.ts](../../frontend/src/lib/api.ts). Locate the `LocalLLMConfig` interface (around line 76) and replace it with:

```typescript
export interface LocalLLMConfig {
    hf_id: string;
    quant: LocalLLMQuant;
    idle_seconds: number;
    gguf_file?: string | null;
}
```

- [ ] **Step 2: Extend CachedModelInfo with gguf fields**

Locate the `CachedModelInfo` interface (right after LocalLLMConfig) and replace with:

```typescript
export interface CachedModelInfo {
    hf_id: string;
    size_bytes: number;
    snapshot_path: string;
    gguf_files?: string[] | null;
    active_gguf_file?: string | null;
}
```

- [ ] **Step 3: Extend EnvConfigPayload with the new env keys**

Locate `EnvConfigPayload` (around line 30). Replace with (adding two new optional fields):

```typescript
export interface EnvConfigPayload {
    DASHSCOPE_API_KEY?: string;
    ALIBABA_CLOUD_ACCESS_KEY_ID?: string;
    ALIBABA_CLOUD_ACCESS_KEY_SECRET?: string;
    OSS_BUCKET_NAME?: string;
    OSS_ENDPOINT?: string;
    OSS_BASE_PATH?: string;
    KLING_PROVIDER_MODE?: ProviderMode;
    VIDU_PROVIDER_MODE?: ProviderMode;
    PIXVERSE_PROVIDER_MODE?: ProviderMode;
    KLING_ACCESS_KEY?: string;
    KLING_SECRET_KEY?: string;
    VIDU_API_KEY?: string;
    LLM_PROVIDER?: "dashscope" | "openai" | "local";
    DASHSCOPE_MODEL?: string;
    LOCAL_LLM_HF_ID?: string;
    LOCAL_LLM_GGUF_FILE?: string;
    endpoint_overrides?: Record<string, string>;
    [key: string]: string | Record<string, string> | undefined;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): extend api types with gguf_file and DASHSCOPE_MODEL"
```

---

## Task 10: Frontend — LLMModelSection component

**Files:**
- Create: `frontend/src/components/common/LLMModelSection.tsx`

- [ ] **Step 1: Create the component**

Create [frontend/src/components/common/LLMModelSection.tsx](../../frontend/src/components/common/LLMModelSection.tsx):

```tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import { Cpu, Check, Plus, Loader2 } from "lucide-react";
import { api, type LocalLLMStatus, type CachedModelInfo, type EnvConfigPayload } from "@/lib/api";

const DEFAULT_LOCAL_HF_ID = "anthfu/Qwen3.6-35B-A3B-APEX-GGUF";

interface CloudModel {
    id: string;
    name: string;
    description: string;
}

const DASHSCOPE_LLM_MODELS: CloudModel[] = [
    { id: "qwen3-max", name: "Qwen3 Max", description: "Latest, strongest" },
    { id: "qwen-plus", name: "Qwen Plus", description: "Default, balanced" },
    { id: "qwen-max", name: "Qwen Max", description: "Long context" },
    { id: "qwen-turbo", name: "Qwen Turbo", description: "Fast & cheap" },
];

function formatBytes(n: number): string {
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function shortHfId(hf_id: string): string {
    // "anthfu/Qwen3.6-35B-A3B-APEX-GGUF" -> "Qwen3.6-35B-A3B-APEX-GGUF"
    const i = hf_id.indexOf("/");
    return i === -1 ? hf_id : hf_id.slice(i + 1);
}

function shortGgufFile(file?: string | null): string | null {
    if (!file) return null;
    // "model-Q4_K_M.gguf" -> "Q4_K_M"
    const m = file.match(/(Q\d+_[A-Z0-9_]+)/);
    return m ? `GGUF ${m[1]}` : "GGUF";
}

export default function LLMModelSection() {
    const [env, setEnv] = useState<EnvConfigPayload | null>(null);
    const [cached, setCached] = useState<CachedModelInfo[]>([]);
    const [status, setStatus] = useState<LocalLLMStatus | null>(null);
    const [busy, setBusy] = useState(false);
    const [addInput, setAddInput] = useState<string | null>(null);  // null = card collapsed; "" = expanded with empty input
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            const [e, c, s] = await Promise.all([
                api.getEnvConfig(),
                api.listCachedLLMs(),
                api.getLocalLLMStatus(),
            ]);
            setEnv(e);
            setCached(c);
            setStatus(s);
        } catch (e) {
            console.error("LLM section refresh failed", e);
        }
    }, []);

    useEffect(() => {
        refresh();
        const id = setInterval(refresh, 3000);
        return () => clearInterval(id);
    }, [refresh]);

    const provider = env?.LLM_PROVIDER ?? "dashscope";
    const dashscopeModel = (env?.DASHSCOPE_MODEL as string) ?? "qwen-plus";
    const localHfId = (env?.LOCAL_LLM_HF_ID as string) ?? "";

    const onPickCloud = async (id: string) => {
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.saveEnvConfig({ LLM_PROVIDER: "dashscope", DASHSCOPE_MODEL: id } as EnvConfigPayload);
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Save failed: ${e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onPickLocal = async (m: CachedModelInfo) => {
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.configureLocalLLM({
                hf_id: m.hf_id,
                quant: "auto",
                idle_seconds: 3,
                gguf_file: m.active_gguf_file ?? null,
            });
            await refresh();
        } catch (e: any) {
            setErrorMsg(`Configure failed: ${e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const onAddSubmit = async () => {
        const hf = (addInput ?? "").trim();
        if (!hf) return;
        setBusy(true);
        setErrorMsg(null);
        try {
            await api.configureLocalLLM({ hf_id: hf, quant: "auto", idle_seconds: 3, gguf_file: null });
            await api.loadLocalLLM();        // triggers download + load (may take many minutes)
            setAddInput(null);
            await refresh();
        } catch (e: any) {
            setErrorMsg(`${e.response?.data?.detail ?? e.message ?? e}`);
        } finally {
            setBusy(false);
        }
    };

    const isDownloading = status?.state === "LOADING" || status?.state === "DOWNLOADING";

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm font-bold text-white">
                <Cpu size={16} className="text-amber-400" />
                <span>LLM (Script Processing)</span>
            </div>

            {/* Cloud cards */}
            <div className="space-y-2">
                <label className="text-xs text-gray-400">Cloud (DashScope)</label>
                <div className="grid grid-cols-2 gap-2">
                    {DASHSCOPE_LLM_MODELS.map((m) => {
                        const selected = provider === "dashscope" && dashscopeModel === m.id;
                        return (
                            <button
                                key={m.id}
                                onClick={() => !busy && onPickCloud(m.id)}
                                disabled={busy}
                                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left disabled:opacity-50 ${
                                    selected
                                        ? "border-green-500/50 bg-green-500/10"
                                        : "border-white/10 hover:border-white/20 bg-white/5"
                                }`}
                            >
                                {selected && (
                                    <div className="absolute top-2 right-2">
                                        <Check size={14} className="text-green-400" />
                                    </div>
                                )}
                                <span className="text-sm font-medium text-white">{m.name}</span>
                                <span className="text-xs text-gray-500">{m.description}</span>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Local cards */}
            <div className="space-y-2">
                <label className="text-xs text-gray-400">Local (HuggingFace)</label>
                <div className="grid grid-cols-2 gap-2">
                    {cached.map((m) => {
                        const selected = provider === "local" && localHfId === m.hf_id;
                        const ggufLabel = shortGgufFile(m.active_gguf_file);
                        return (
                            <button
                                key={m.hf_id}
                                onClick={() => !busy && onPickLocal(m)}
                                disabled={busy}
                                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left disabled:opacity-50 ${
                                    selected
                                        ? "border-purple-500/50 bg-purple-500/10"
                                        : "border-white/10 hover:border-white/20 bg-white/5"
                                }`}
                            >
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                                        Local
                                    </span>
                                    {ggufLabel && (
                                        <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
                                            {ggufLabel}
                                        </span>
                                    )}
                                </div>
                                {selected && (
                                    <div className="absolute top-2 right-2">
                                        <Check size={14} className="text-purple-400" />
                                    </div>
                                )}
                                <span className="text-sm font-medium text-white" title={m.hf_id}>
                                    {shortHfId(m.hf_id)}
                                </span>
                                <span className="text-xs text-gray-500">{formatBytes(m.size_bytes)}</span>
                            </button>
                        );
                    })}

                    {/* Add Local Model card */}
                    {addInput === null ? (
                        <button
                            onClick={() => setAddInput(DEFAULT_LOCAL_HF_ID)}
                            disabled={busy}
                            className="flex flex-col items-center justify-center p-3 rounded-lg border border-dashed border-white/20 hover:border-white/40 bg-white/5 transition-all text-gray-400 hover:text-white disabled:opacity-50"
                        >
                            <Plus size={20} className="mb-1" />
                            <span className="text-sm">Add Local Model</span>
                        </button>
                    ) : (
                        <div className="col-span-2 p-3 rounded-lg border border-amber-500/30 bg-amber-500/5 space-y-2">
                            <div className="text-xs text-gray-400">HuggingFace Model ID</div>
                            <input
                                type="text"
                                value={addInput}
                                onChange={(e) => setAddInput(e.target.value)}
                                placeholder="org/model-name"
                                disabled={busy}
                                className="w-full rounded bg-black/40 border border-white/10 px-3 py-2 text-sm text-white"
                            />
                            <div className="flex justify-end gap-2 pt-1">
                                <button
                                    onClick={() => { setAddInput(null); setErrorMsg(null); }}
                                    disabled={busy}
                                    className="px-3 py-1.5 text-xs text-gray-400 hover:text-white"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={onAddSubmit}
                                    disabled={busy || !addInput.trim()}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-amber-600 hover:bg-amber-500 text-white rounded disabled:opacity-50"
                                >
                                    {busy && isDownloading ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin" />
                                            Downloading...
                                        </>
                                    ) : busy ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin" />
                                            Working...
                                        </>
                                    ) : (
                                        "Download & Use"
                                    )}
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {errorMsg && (
                <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded px-3 py-2">
                    {errorMsg}
                </div>
            )}
        </div>
    );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/common/LLMModelSection.tsx
git commit -m "feat(frontend): add LLMModelSection with cloud cards, dynamic local cards, and add-model flow"
```

---

## Task 11: Frontend — embed LLMModelSection in ModelSettingsModal

**Files:**
- Modify: `frontend/src/components/common/ModelSettingsModal.tsx`

- [ ] **Step 1: Import the new section**

Edit [frontend/src/components/common/ModelSettingsModal.tsx](../../frontend/src/components/common/ModelSettingsModal.tsx). At the top of the file, find the existing `lucide-react` import and add `LLMModelSection`:

```tsx
import LLMModelSection from './LLMModelSection';
```

(Place it on its own line right after the `import { api } from '@/lib/api';` line.)

- [ ] **Step 2: Embed the section in the modal content**

Locate the existing Motion section's closing `</div>` (right before `{/* Footer */}`). The structure is:

```tsx
                        {/* Motion Section */}
                        <div className="space-y-4">
                            ...
                        </div>
                    </div>

                    {/* Footer */}
```

After the Motion section's closing `</div>` (and before the outer `</div>` that closes the content area), add a divider and the new section:

```tsx
                        <div className="border-t border-white/10" />

                        {/* LLM Section */}
                        <LLMModelSection />
```

The final structure should look like:

```tsx
                        {/* Motion Section */}
                        <div className="space-y-4">
                            ...existing motion content...
                        </div>

                        <div className="border-t border-white/10" />

                        {/* LLM Section */}
                        <LLMModelSection />
                    </div>

                    {/* Footer */}
```

- [ ] **Step 3: Manual verify**

Open http://localhost:3000, go to a project, click the gear/settings icon to open Generation Settings. Scroll to the bottom — should see "🟡 LLM (Script Processing)" section with 4 cloud cards and an "+ Add Local Model" card (plus any cached local cards if present).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/common/ModelSettingsModal.tsx
git commit -m "feat(frontend): embed LLMModelSection in ModelSettingsModal"
```

---

## Task 12: End-to-end smoke (cloud + small local + cached card)

**Files:** (no code changes; manual verification)

- [ ] **Step 1: Cloud card works**

In the Generation Settings modal, click "Qwen Max" cloud card. Verify:
- Card gains green check
- Other cloud cards lose check
- `.env` updated:

```bash
grep -E "^(LLM_PROVIDER|DASHSCOPE_MODEL)" /s/AI/lumenxDev/lumenx/.env
```

Expected:
```
LLM_PROVIDER='dashscope'
DASHSCOPE_MODEL='qwen-max'
```

- [ ] **Step 2: Local card (small, already cached) works**

If `Qwen/Qwen2.5-0.5B-Instruct` is in cached list (from prior test runs), click it. Verify:
- Card gains purple check; cloud loses check
- `.env` shows `LLM_PROVIDER='local'` and `LOCAL_LLM_HF_ID='Qwen/Qwen2.5-0.5B-Instruct'`

If no local cached model exists, skip to Step 3 first.

- [ ] **Step 3: Add Local Model with a small GGUF (~400MB)**

Click "+ Add Local Model". Input is pre-filled with `anthfu/Qwen3.6-35B-A3B-APEX-GGUF` — **change it** to a smaller test model:

```
Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

Click "Download & Use". Watch the button text become "Downloading...". Backend log will show:

```
[backend] INFO ... Loading GGUF Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf (n_ctx=8192)
[backend] INFO ... Loaded GGUF Qwen/... in <N>s
```

Expected: ~400MB download (1-3 min), then load completes (<10s). New card appears in local section with `LOCAL` and `GGUF Q4_K_M` badges.

- [ ] **Step 4: Cached GGUF card click selects it**

Click the new GGUF card. Verify it gains the purple check and `.env` shows:

```
LLM_PROVIDER='local'
LOCAL_LLM_HF_ID='Qwen/Qwen2.5-0.5B-Instruct-GGUF'
LOCAL_LLM_GGUF_FILE='qwen2.5-0.5b-instruct-q4_k_m.gguf'
```

- [ ] **Step 5: Verify the model actually responds**

In another terminal:

```bash
curl -s -X POST http://localhost:17177/llm/local/test | python -m json.tool
```

Expected: `{"ok": true, "elapsed_sec": <N>, "response": "<some greeting>"}`

- [ ] **Step 6: Verify auto-unload still works for GGUF**

Wait 5 seconds without doing anything. Then:

```bash
curl -s http://localhost:17177/llm/local/status | python -m json.tool
```

Expected: `"state": "UNLOADED"`.

---

## Task 13: Run full test suite

- [ ] **Step 1: Run all llm_local tests**

```bash
cd /s/AI/lumenxDev/lumenx && .venv/Scripts/python.exe -m pytest tests/llm_local/ -v
```

Expected: all tests pass. Approximate counts after this plan:
- test_vram.py: 13
- test_config.py: 9
- test_manager.py: 11
- test_api.py: 11
- test_runtime.py: 4 (skipped without GPU)
- test_runtime_gguf.py: 3 (skipped without GPU)
- test_gguf_utils.py: 9
- test_llm_adapter.py: 3

Total: ~63 tests (~56 unit + 7 GPU integration).

- [ ] **Step 2: If any test fails, fix it inline and commit**

```bash
git add tests/
git commit -m "test: stabilize llm_local test suite after picker + gguf changes"
```

---

## Done

After this plan completes:
- The Generation Settings modal has an "LLM (Script Processing)" section with cloud + local cards.
- "+ Add Local Model" downloads any HF id (transformers safetensors **or** GGUF) and activates it.
- GGUF repos auto-route through `llama-cpp-python` with quant auto-pick (Q4_K_M priority).
- DashScope cloud model is selectable via card click (writes `DASHSCOPE_MODEL` to `.env`, read by `LLMAdapter._get_default_model()`).
- All existing local-LLM features (idle unload, OOM step-down for transformers, etc.) keep working.

The default suggested model in the "+ Add" input is `anthfu/Qwen3.6-35B-A3B-APEX-GGUF` (~20GB). User can edit before downloading.
