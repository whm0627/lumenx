# LLM Model Picker (in-modal) + GGUF Runtime — Design

**Date:** 2026-04-25
**Status:** Approved (brainstorming)
**Scope:** Two coupled features shipping together. Builds on the prior local-LLM-runtime work.
**Reference (prior spec):** [2026-04-25-local-llm-runtime-design.md](./2026-04-25-local-llm-runtime-design.md)

---

## 1. Goal

Two related additions on top of the existing local-LLM runtime:

**A. In-modal LLM picker.** Add an "LLM (Script Processing)" section to the per-project Generation Settings dialog (`ModelSettingsModal`) — same card-style UI as the existing T2I / I2I / I2V sections. The picker is **global** (writes `.env`, not project state) but lives next to project model settings for ergonomics. Cards include cloud (DashScope) and locally-cached HuggingFace models. A "+ Add Local Model" card lets the user type any HF id and trigger one-shot download + activation.

**B. GGUF runtime backend.** Extend `ModelManager` to dispatch GGUF repos to a new `GGUFRuntime` (backed by `llama-cpp-python`) instead of the existing transformers `LocalRuntime`. Required for the user's chosen default model `anthfu/Qwen3.6-35B-A3B-APEX-GGUF` (Qwen3 MoE 35B/3B-active in GGUF Q4_K_M).

User experience after this ships:

1. User opens any project → clicks Generation Settings → sees LLM section with cloud + cached-local cards.
2. Click cloud card → instant `LLM_PROVIDER=dashscope` + `DASHSCOPE_MODEL=qwen-plus`. Click local card → instant `LLM_PROVIDER=local` + `LOCAL_LLM_HF_ID=...`.
3. Click "+ Add Local Model" → input pre-filled with `anthfu/Qwen3.6-35B-A3B-APEX-GGUF`. Click Download → backend recognises GGUF, picks `Q4_K_M` variant, downloads (~20GB), loads via `llama_cpp`, becomes active.
4. After 3s idle → unloaded, VRAM freed (same idle policy applies to both runtimes).

## 2. Non-Goals

- Per-project LLM selection (explicitly chosen Global in brainstorming Q1)
- OpenAI-compatible cards in the modal (configuration is too multi-field; stays in SettingsPage)
- llama.cpp model **conversion** (we only consume pre-converted GGUF files from HF)
- Manual GGUF quant override in the modal UI (auto-picked at add-time; user can change via SettingsPage in a follow-up)
- Multi-GPU / tensor-parallel inference for GGUF (single-GPU only)
- VRAM tracking parity between transformers and GGUF runtime (GGUF reports 0 in v1; nvidia-ml-py polling deferred)
- Streaming responses (still out of scope per prior spec)

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ ModelSettingsModal (per-project)                                │
│  ├ Assets / Storyboard / Motion sections (existing, project)    │
│  └ LLM (Script Processing) section (NEW, GLOBAL)                │
│         │                                                        │
│         ├─ Cloud cards: Qwen3 Max / Plus / Max-LongCtx / Turbo  │
│         ├─ Local cards: dynamic from /llm/local/cached          │
│         │   - "LOCAL" badge top-right                           │
│         │   - GGUF badge "GGUF Q4_K_M" if applicable            │
│         └─ "+ Add Local Model" card                             │
│             - Inline input + Download & Use button              │
│             - Default placeholder = anthfu/Qwen3.6-35B-A3B-APEX-GGUF │
└────────────────────────────────────────────────────────────────┘
                          │ HTTP
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ Backend                                                         │
│  POST /llm/local/configure  { hf_id, quant, idle_seconds,       │
│                                gguf_file? }   ← NEW field        │
│  POST /llm/local/load                                           │
│  GET  /llm/local/cached     → now reports gguf_file when applicable │
│  POST /config/env           { LLM_PROVIDER, DASHSCOPE_MODEL,    │
│                                LOCAL_LLM_HF_ID, ... }            │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ LLMAdapter (extended)                                           │
│   provider == "dashscope"  → uses DASHSCOPE_MODEL env (NEW)     │
│   provider == "openai"     → unchanged                          │
│   provider == "local"      → ModelManager.chat_sync()           │
└────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────┐
│ ModelManager (extended)                                         │
│   _runtime_factory now dispatches:                              │
│     is_gguf_repo(hf_id)?                                        │
│       yes → GGUFRuntime(hf_id, gguf_file)                       │
│       no  → LocalRuntime(hf_id, quant)   (existing)             │
└────────────────────────────────────────────────────────────────┘
                  │                              │
                  ▼                              ▼
       ┌─────────────────┐           ┌──────────────────────┐
       │ LocalRuntime    │           │ GGUFRuntime (NEW)    │
       │  transformers   │           │  llama-cpp-python    │
       │  + bnb quant    │           │  Llama(model_path,   │
       │  + chat template│           │        n_gpu_layers, │
       │                 │           │        chat_format)  │
       └─────────────────┘           └──────────────────────┘
                  │                              │
                  └──────────┬───────────────────┘
                             ▼
                ┌─────────────────────────────┐
                │ output/models/LLM/hub/      │
                │   models--<org>--<name>/    │
                │     snapshots/<rev>/...     │
                └─────────────────────────────┘
```

## 4. File Layout

**New files:**

```
src/llm_local/
├── runtime_gguf.py          # GGUFRuntime: llama_cpp wrapper
└── gguf_utils.py            # is_gguf_repo(), pick_default_gguf_file(), list_gguf_files()

frontend/src/components/common/
└── LLMModelSection.tsx      # The new section embedded into ModelSettingsModal
```

**Modified files:**

```
src/llm_local/manager.py                # factory dispatch on GGUF vs not; pass gguf_file
src/llm_local/runtime.py                # no change to interface; minor: declare LoadResult is shared
src/llm_local/config.py                 # add gguf_file: Optional[str] field
src/llm_local/api.py                    # configure endpoint accepts gguf_file; cached endpoint reports gguf_file
src/apps/comic_gen/llm_adapter.py       # _get_default_model() reads DASHSCOPE_MODEL
requirements.txt                        # add llama-cpp-python (CUDA build)

frontend/src/components/common/ModelSettingsModal.tsx   # embed <LLMModelSection />
frontend/src/lib/api.ts                                  # add DASHSCOPE_MODEL to EnvConfigPayload type; LocalLLMConfig adds gguf_file?
.env.example                                             # document DASHSCOPE_MODEL and LOCAL_LLM_GGUF_FILE
```

**Deliberately NOT modified:**

- `src/llm_local/__init__.py` — HF_HOME setup unchanged
- `src/llm_local/vram.py` — VRAM detection still used for transformers path; GGUF doesn't auto-quant
- `frontend/src/components/settings/SettingsPage.tsx` and `LocalLLMPanel.tsx` — kept as-is. The new modal section is a parallel UI for the same global state; both reflect each other through `/config/env` and `/llm/local/status`.

## 5. Components

### 5.1 `gguf_utils.py` — GGUF detection helpers

```python
def is_gguf_repo(hf_id: str) -> bool:
    """Cheap heuristic: hf_id contains 'GGUF' (case-insensitive)."""
    return "gguf" in hf_id.lower()

def list_gguf_files(hf_id: str) -> list[str]:
    """Network call: list .gguf files in repo via huggingface_hub.HfApi."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(hf_id)
    return [f for f in files if f.endswith(".gguf")]

# Quant priority (best balance for typical 24GB VRAM): Q4_K_M > Q5_K_M > Q4_K_S > Q6_K > Q8_0 > others
_QUANT_PRIORITY = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0"]

def pick_default_gguf_file(files: list[str]) -> Optional[str]:
    """Pick the highest-priority quant from the list. None if no .gguf files."""
    for quant in _QUANT_PRIORITY:
        for f in files:
            if quant in f:
                return f
    return files[0] if files else None
```

### 5.2 `runtime_gguf.py` — GGUFRuntime

Same interface as `LocalRuntime` — drop-in for ModelManager. Implementation uses `llama_cpp.Llama`:

```python
class GGUFRuntime:
    def __init__(self, hf_id: str, gguf_file: str):
        self.hf_id = hf_id
        self.gguf_file = gguf_file
        self.quant = QuantMode.AUTO  # not meaningful for GGUF
        self._llama: Optional[Llama] = None
        self._loaded = False

    def load(self) -> LoadResult:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        t0 = time.monotonic()
        local_path = hf_hub_download(repo_id=self.hf_id, filename=self.gguf_file)

        self._llama = Llama(
            model_path=local_path,
            n_gpu_layers=-1,           # offload all layers to GPU
            n_ctx=8192,                # context window; reasonable default
            verbose=False,
            chat_format=None,          # auto-detect from GGUF metadata
        )
        self._loaded = True
        return LoadResult(
            hf_id=self.hf_id,
            quant_mode=QuantMode.AUTO,  # unused for GGUF
            vram_used_mb=0,             # llama_cpp doesn't expose this; reported as 0
            elapsed_sec=time.monotonic() - t0,
        )

    def chat(self, messages, max_new_tokens=1024, temperature=0.7) -> str:
        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"]

    def unload(self) -> None:
        if self._llama is not None:
            del self._llama
            self._llama = None
        self._loaded = False
        gc.collect()
```

Note: `n_ctx=8192` is hardcoded for v1. Most script-processing prompts fit comfortably. Configurable later if needed.

### 5.3 `manager.py` — Factory dispatch

```python
def _default_runtime_factory(hf_id, quant, gguf_file=None):
    if is_gguf_repo(hf_id):
        if gguf_file is None:
            gguf_file = pick_default_gguf_file(list_gguf_files(hf_id))
            if gguf_file is None:
                raise RuntimeError(f"No .gguf files found in {hf_id}")
        return GGUFRuntime(hf_id=hf_id, gguf_file=gguf_file)
    return LocalRuntime(hf_id=hf_id, quant=quant)
```

The factory signature changes — `RuntimeFactory` now takes an optional `gguf_file`. Existing tests (which pass a stub factory) continue to work because Python doesn't enforce the signature beyond call-time.

`_load_locked` now passes the resolved `gguf_file` to the factory. OOM step-down is a no-op for GGUF (quant baked into file); ModelManager skips the step-down loop for GGUF and propagates errors directly.

### 5.4 `config.py` — LocalLLMConfig

Add one field:

```python
class LocalLLMConfig(BaseModel):
    hf_id: str = Field(default="")
    quant: QuantMode = Field(default=QuantMode.AUTO)
    idle_seconds: int = Field(default=3, ge=1, le=3600)
    gguf_file: Optional[str] = Field(default=None,
        description="Specific .gguf file in a GGUF repo; auto-picked if omitted")
```

`from_env` reads `LOCAL_LLM_GGUF_FILE`; `to_env_dict` emits it.

### 5.5 `api.py` — `/llm/local/*` endpoints

`/llm/local/configure` request payload gains `gguf_file: Optional[str]`. If omitted and the model is GGUF, the manager auto-picks at load time.

`/llm/local/cached` response items gain an optional `gguf_file: str | None` (the currently-configured file for that cached repo, or null).

### 5.6 `llm_adapter.py` — DASHSCOPE_MODEL support

```python
def _get_default_model(self) -> str:
    if self.provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    return os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")  # was hardcoded
```

One-line change. Backward-compatible (falls back to current default).

### 5.7 Frontend `LLMModelSection.tsx`

```
┌─ LLM (Script Processing) ───────────────────────────────────────┐
│                                                                  │
│ Cloud (DashScope)                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐                │
│  │ Qwen3 Max           │  │ Qwen Plus       ✓   │                │
│  │ Latest, strongest   │  │ Default, balanced   │                │
│  └─────────────────────┘  └─────────────────────┘                │
│  ┌─────────────────────┐  ┌─────────────────────┐                │
│  │ Qwen Max            │  │ Qwen Turbo          │                │
│  │ Long context        │  │ Fast & cheap        │                │
│  └─────────────────────┘  └─────────────────────┘                │
│                                                                  │
│ Local (HuggingFace)                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                │
│  │  LOCAL              │  │  LOCAL  GGUF Q4_K_M │                │
│  │ Qwen2.5-0.5B-Instr  │  │ Qwen3.6-35B-A3B-APEX│                │
│  │ 953 MB              │  │ 19.8 GB             │                │
│  └─────────────────────┘  └─────────────────────┘                │
│  ┌─────────────────────────────────────────────────┐             │
│  │ + Add Local Model                                │             │
│  │   anthfu/Qwen3.6-35B-A3B-APEX-GGUF              │             │
│  │   [Cancel]            [Download & Use]           │             │
│  └─────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Component contract:**
- Self-contained: fetches own state from `/config/env` (current selection), `/llm/local/cached` (local card list), polls `/llm/local/status` while a download is in progress
- Stateless from the parent's view (parent just renders `<LLMModelSection />`)
- Uses the same card visual style as other sections (border, hover, check icon when selected)
- Cloud cards use the `green` accent (matching Assets section)
- Local cards use the `purple` accent (matching Motion section, since both involve heavier compute)
- "LOCAL" badge: small `bg-purple-500/20 text-purple-300` pill in card top-right
- "GGUF Q4_K_M" badge: smaller `bg-amber-500/20 text-amber-300` pill, rendered next to LOCAL when applicable

**Selected-card detection:**
- Cloud: `LLM_PROVIDER === "dashscope" && DASHSCOPE_MODEL === card.id` (or both empty → first card "Qwen Plus" is implicit default)
- Local: `LLM_PROVIDER === "local" && LOCAL_LLM_HF_ID === card.hf_id`

**Click handlers:**
- Cloud card click → `await api.saveEnvConfig({ LLM_PROVIDER: "dashscope", DASHSCOPE_MODEL: id })` → refresh
- Local card click → `await api.configureLocalLLM({ hf_id, quant: "auto", idle_seconds: 3 })` → refresh
- "+ Add" expand → input pre-filled with `anthfu/Qwen3.6-35B-A3B-APEX-GGUF`
- "Download & Use" click → `configureLocalLLM` then `loadLocalLLM` (with timeout 1 hour for big GGUF), poll `/llm/local/status`, refresh `/llm/local/cached`

## 6. API Contracts (changes only)

### `POST /llm/local/configure` (extended)

```json
// Request
{
  "hf_id": "anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
  "quant": "auto",            // ignored for GGUF
  "idle_seconds": 3,
  "gguf_file": null            // NEW: explicit file or null for auto-pick
}
// Response: unchanged
{ "ok": true, "persisted": true }
```

When `gguf_file` is null and `is_gguf_repo(hf_id)` is true, the auto-pick happens lazily inside ModelManager._load_locked at first load (network call to `list_repo_files`).

### `GET /llm/local/cached` (extended response)

```json
[
  {
    "hf_id": "anthfu/Qwen3.6-35B-A3B-APEX-GGUF",
    "size_bytes": 21274836480,
    "snapshot_path": "...",
    "gguf_files": ["model-Q4_K_M.gguf"],     // NEW: present for GGUF repos
    "active_gguf_file": "model-Q4_K_M.gguf"  // NEW: which one is currently selected
  },
  {
    "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "size_bytes": 999586347,
    "snapshot_path": "...",
    "gguf_files": null,                       // NEW: null for non-GGUF
    "active_gguf_file": null
  }
]
```

### `GET /config/env` and `POST /config/env`

No schema change required (already a flexible dict). Frontend adds `DASHSCOPE_MODEL` and `LOCAL_LLM_GGUF_FILE` as keys.

## 7. Data Flow Examples

### Adding the default GGUF model from scratch

```
t=0    User opens Generation Settings, scrolls to LLM section.
t=0    LLMModelSection mounts, fetches /config/env + /llm/local/cached.
       Empty local cards. Cloud cards show Qwen Plus selected (default).
t=2    User clicks "+ Add Local Model" → input expands, pre-filled with
       "anthfu/Qwen3.6-35B-A3B-APEX-GGUF".
t=3    User clicks "Download & Use".
t=3    POST /llm/local/configure { hf_id, quant:"auto", idle_seconds:3, gguf_file:null }
       → persists LLM_PROVIDER=local + LOCAL_LLM_HF_ID + LOCAL_LLM_GGUF_FILE=null
t=3    POST /llm/local/load → ModelManager._load_locked():
         is_gguf_repo("anthfu/...GGUF") → true
         list_repo_files() → [".gitattributes", "README.md", "model-Q4_K_M.gguf"]
         pick_default_gguf_file() → "model-Q4_K_M.gguf"
         hf_hub_download(repo_id, "model-Q4_K_M.gguf") → ~20GB download (5-30 min)
         Llama(model_path, n_gpu_layers=-1, n_ctx=8192) → load to GPU
         state = READY
       Frontend polls /llm/local/status during this; shows "DOWNLOADING / LOADING".
t=600  state = READY. Card replaces input form. Selected.
t=603  Idle 3s → unloaded.
```

### Switching from cloud to a cached local model

```
Current: LLM_PROVIDER=dashscope, DASHSCOPE_MODEL=qwen-plus
User clicks the cached "Qwen2.5-0.5B-Instruct" card.
Frontend: configureLocalLLM({ hf_id: "Qwen/Qwen2.5-0.5B-Instruct", quant:"auto", idle_seconds:3 })
Backend persists LLM_PROVIDER=local + LOCAL_LLM_HF_ID. ModelManager.configure() unloads (no-op since UNLOADED).
Cloud card "Qwen Plus" loses ✓; local "Qwen2.5..." gains ✓.
Next chat call → lazy-loads via LocalRuntime (transformers path).
```

### Switching from one cloud model to another

```
Current: LLM_PROVIDER=dashscope, DASHSCOPE_MODEL=qwen-plus
User clicks "Qwen3 Max" card.
Frontend: saveEnvConfig({ LLM_PROVIDER:"dashscope", DASHSCOPE_MODEL:"qwen3-max" })
Backend persists. No model loading. Next LLMAdapter.chat() reads env again,
reads "qwen3-max", uses it.
```

## 8. Error Handling

| Scenario | Handling |
|----------|----------|
| HF id not GGUF but matches the heuristic (false positive) | `list_gguf_files()` returns []; `pick_default_gguf_file()` returns None; load raises `"No .gguf files found in <hf_id>"`. User can rename or use SettingsPage to set explicit `LOCAL_LLM_GGUF_FILE`. |
| GGUF repo with multi-quant files, none match `_QUANT_PRIORITY` | `pick_default_gguf_file` falls back to first .gguf file. Logged. |
| llama-cpp-python not installed | `from llama_cpp import Llama` ImportError → propagate with install instructions. |
| GGUF download fails (network) | `hf_hub_download` raises; ModelManager state → ERROR. UI shows error. Retry by re-clicking. |
| Llama() load fails (file corrupt / OOM) | Exception → ERROR. For OOM specifically: GGUF cannot step-down; we surface the message and suggest a smaller .gguf file via SettingsPage. |
| User enters bad HF id in "+ Add" | configure persists empty; load fails; error message displayed. Card stays in error state until user retries or cancels. |
| Cloud DASHSCOPE_MODEL set to a model name DashScope rejects | Existing LLMAdapter error handling — `RuntimeError("DashScope API error: ...")` propagates to caller. UI is the same as any DashScope failure. |

## 9. Dependencies

`requirements.txt` adds:

```
llama-cpp-python>=0.3.0
```

**Install**: For CUDA acceleration on Windows + Python 3.12, use the prebuilt CUDA 12.1 wheel:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

If the index doesn't have a matching wheel, fallback to the CPU-only PyPI build (slower, but works without compilation).

`huggingface_hub` is already installed (transitively via transformers); we just use `hf_hub_download` and `HfApi.list_repo_files` from it.

## 10. Persistence

Three new `.env` keys:

```
DASHSCOPE_MODEL=qwen-plus
LOCAL_LLM_GGUF_FILE=model-Q4_K_M.gguf
```

`LOCAL_LLM_GGUF_FILE` may be empty/unset; auto-picked at first load.
`DASHSCOPE_MODEL` may be empty/unset; falls back to the existing hardcoded default in `_get_default_model()`.

## 11. Testing

**Unit tests (no real model):**
- `tests/llm_local/test_gguf_utils.py`:
  - `is_gguf_repo` true/false cases
  - `pick_default_gguf_file` priority order, fallback, empty
- `tests/llm_local/test_manager.py` (extend existing):
  - Factory dispatches GGUF id → GGUFRuntime, non-GGUF → LocalRuntime (using stub factory + monkeypatched `is_gguf_repo`)
  - configure() with gguf_file persists it
- `tests/llm_local/test_config.py` (extend):
  - `LocalLLMConfig.from_env` reads `LOCAL_LLM_GGUF_FILE`
  - `to_env_dict` emits it
- `tests/llm_local/test_api.py` (extend):
  - configure endpoint accepts gguf_file
  - cached endpoint reports `gguf_files` and `active_gguf_file`
- `tests/llm_local/test_llm_adapter.py` (NEW small file):
  - `_get_default_model()` reads `DASHSCOPE_MODEL` env var when set
  - falls back to default when unset

**Integration test (real GGUF, optional / manual):**
- `tests/llm_local/test_runtime_gguf.py` — load a tiny GGUF (e.g., `bartowski/Qwen2.5-0.5B-Instruct-GGUF` Q4_K_M ≈ 400MB), do one chat, unload. Marked `slow`, skipped without GPU.

**No automated test for the 35B default model** (too large; covered by manual e2e).

## 12. Risks

| Risk | Mitigation |
|------|------------|
| `llama-cpp-python` install fails on Windows (no prebuilt CUDA wheel for some Python/CUDA combos) | Document the `--extra-index-url` cu121 path. Fallback to CPU build if needed. |
| `Llama` constructor signature varies across versions | Pin `llama-cpp-python>=0.3.0` (stable since late 2024). Test with installed version before completing. |
| GGUF chat template not in metadata for some community models | `Llama` exposes a `chat_format` arg with built-in templates ("qwen", "chatml", etc.). Fallback in v2; for v1 we trust the model metadata. |
| 35B-A3B GGUF needs ≈ 14-18GB VRAM (offloaded) — might OOM on 24GB if other GPU work running | Surface load errors clearly. User can pick smaller quant via SettingsPage. |
| llama-cpp `n_ctx=8192` hardcoded; long scripts truncate | Prompts in lumenx are usually short (entity extraction is per-scene). Document as v1 limit. |
| Dual runtime drift — two implementations of load/chat/unload | Keep the LoadResult dataclass shared. Both runtimes implement the same minimal interface (no abstract base class to start; YAGNI). |
| HfApi list_repo_files network failure during auto-pick | Fail clearly with retry. User can supply explicit `gguf_file` in SettingsPage to bypass. |

## 13. Out-of-Scope Future Phases (informational)

- Quant file picker UI (dropdown when multiple GGUF quants exist in a repo)
- VRAM tracking via `pynvml` for GGUF runtime
- Per-task model selection (small for entity extraction, big for prompt polish)
- Streaming chat responses
- llama-cpp `n_ctx` configurable per model
- Image / Video model managers in `output/models/{Image,Video}/` (same architectural pattern)
