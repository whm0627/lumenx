# Local LLM Runtime Management — Design

**Date:** 2026-04-25
**Status:** Approved (brainstorming)
**Scope:** Phase 1 — LLM only. Image / Video model management follows the same pattern in later phases.

---

## 1. Goal

Let LumenX run script-processing LLM tasks (entity extraction, style analysis, storyboard analysis, prompt polishing) using a **locally-loaded HuggingFace model**, with the application itself owning the model lifecycle (download → load → use → unload).

User experience:
1. User picks `Local` as the LLM provider in Settings.
2. User types an HF model id (e.g., `Qwen/Qwen3-8B-Instruct`).
3. First call to any LLM-using feature lazily downloads (if missing) + loads the model.
4. After 3 seconds of no LLM activity, the model is automatically unloaded and VRAM is freed.
5. Next call cold-loads again.

## 2. Non-Goals

- Multi-model concurrent loading (24GB VRAM target — one model at a time)
- Streaming responses (existing `LLMAdapter.chat()` returns full string; keep contract)
- LoRA / fine-tuning / training
- Image and Video model management (planned for later phases under the same `output/models/` layout, but **out of scope for this spec**)
- Custom inference parameters per task (temperature, top_p tuning UI) — fixed defaults for now
- "Keep loaded" override / pinning — explicitly chose simplest behavior (option A in brainstorming)

## 3. High-Level Architecture

```
Frontend SettingsPage
  └─ LocalLLMPanel (NEW)
        │ HTTP
        ▼
Backend FastAPI
  ├─ /llm/local/configure   (POST)
  ├─ /llm/local/load        (POST)
  ├─ /llm/local/unload      (POST)
  ├─ /llm/local/status      (GET)
  ├─ /llm/local/test        (POST)
  └─ /llm/local/cached      (GET / DELETE)
        │
        ▼
LLMAdapter (existing, extended)
  provider ∈ {dashscope, openai, local}
  └─ local branch → ModelManager.chat()
        │
        ▼
ModelManager (NEW, asyncio singleton)
  ├─ State machine + asyncio.Lock
  ├─ idle_watcher background task
  └─ delegates loading/inference to ↓
        │
        ▼
LocalRuntime (NEW, transformers wrapper)
  ├─ AutoTokenizer / AutoModelForCausalLM
  ├─ device_map="auto", dtype/quant per VRAM
  └─ apply_chat_template + generate
        │
        ▼
Storage
  HF_HOME = <project>/output/models/LLM/
```

## 4. File Layout

**New files:**

```
src/llm_local/
├── __init__.py
├── manager.py        # ModelManager singleton, state machine, idle watcher
├── runtime.py        # LocalRuntime: transformers load/chat/unload
├── config.py         # Pydantic settings (HF id, quant, idle sec, cache dir)
├── api.py            # FastAPI APIRouter for /llm/local/*
└── vram.py           # VRAM detection + auto quant strategy

frontend/src/components/settings/
└── LocalLLMPanel.tsx # New panel embedded into SettingsPage
```

**Modified files:**

```
src/apps/comic_gen/llm_adapter.py   # add provider == "local" branch
src/apps/comic_gen/api.py           # include_router(local_llm.router)
src/apps/comic_gen/llm.py           # strip response_format kwarg when provider=local
frontend/src/components/settings/SettingsPage.tsx  # add LLM Provider section + embed LocalLLMPanel
frontend/src/lib/api.ts             # client functions for /llm/local/*
requirements.txt                    # add torch, transformers, accelerate, bitsandbytes, huggingface_hub
```

## 5. Components

### 5.1 ModelManager (`src/llm_local/manager.py`)

Asyncio singleton that owns the model's lifecycle. All access is serialized through a single `asyncio.Lock`.

**State machine:**

```
                  configure(hf_id, quant)
                          │
                          ▼
   ┌─────────────────► UNLOADED
   │                      │ load() or first chat()
   │                      ▼
   │                 DOWNLOADING ──(cached)──► LOADING
   │                      │                       │
   │ unload() / idle      │ HF download error     │ OOM / load error
   │                      ▼                       ▼
   ├──────────────────  ERROR ◄───────────────────┘
   │                      │
   │                      │ retry on next chat()
   │                      ▼
   └─────────────  READY ◄──── LOADING (success)
                    │
                    │ chat() (resets last_used_ts)
                    ▲────┘
```

**Public API (async, all acquire lock):**

| Method | Purpose |
|--------|---------|
| `configure(hf_id, quant_mode, idle_sec)` | Set target model. If a different model is already loaded → unload first. |
| `load() -> None` | Force load now (used by explicit "Load" button). Idempotent if already READY for same id. |
| `chat(messages: list[dict], **kwargs) -> str` | Async. Lazy-load if UNLOADED. Update `last_used_ts`. Returns full response string. |
| `chat_sync(messages: list[dict], **kwargs) -> str` | Sync wrapper used by `LLMAdapter`. Bridges via `asyncio.run_coroutine_threadsafe(self.chat(...), loop)` against the FastAPI event loop captured at app startup. |
| `unload() -> None` | Free VRAM. No-op if already UNLOADED. |
| `status() -> StatusDict` | `{state, hf_id, quant_mode, vram_used_mb, vram_total_mb, last_used_ts, error}` |

**Idle watcher:**

- Background `asyncio.Task` started in FastAPI lifespan.
- Wakes every 1s. If state == READY and `now - last_used_ts > idle_sec` → call `unload()`.
- Cancels on app shutdown.

**Concurrency rules:**

- All public methods acquire the same lock — this serializes downloads, loads, inference, and unloads.
- During an in-flight `chat()`, the idle watcher will wait for the lock before unloading. So `chat()` always completes; it just may unload immediately after if idle threshold passed.

### 5.2 LocalRuntime (`src/llm_local/runtime.py`)

Pure-sync transformers wrapper. Knows nothing about state or locking. Run on a thread executor from `ModelManager` so it doesn't block the event loop.

```python
class LocalRuntime:
    def __init__(self, hf_id: str, quant: QuantMode, cache_dir: Path): ...
    def load(self) -> LoadResult: ...        # blocking, returns metadata (vram, params, dtype)
    def chat(self, messages: list[dict], max_new_tokens: int = 1024) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # generate, decode, return only the assistant's new tokens
    def unload(self) -> None:
        del self.model; del self.tokenizer
        torch.cuda.empty_cache(); gc.collect()
```

Loading uses:
```python
AutoModelForCausalLM.from_pretrained(
    hf_id,
    torch_dtype=torch.bfloat16,        # or auto-picked
    device_map="auto",
    quantization_config=...,            # only if 8/4-bit
    cache_dir=str(cache_dir),
    low_cpu_mem_usage=True,
)
```

### 5.3 Quant strategy (`src/llm_local/vram.py`)

On first call after configure, detect VRAM via `torch.cuda.mem_get_info()`. With user's 24GB target, defaults are:

| Param count (from HF config) | Quant default | Est. VRAM |
|------------------------------|---------------|-----------|
| ≤ 4B                         | bfloat16      | ~8 GB     |
| 7B–8B                        | bfloat16      | ~16 GB    |
| 13B–14B                      | bnb 8-bit     | ~14 GB    |
| 32B                          | bnb 4-bit     | ~18 GB    |
| > 32B                        | reject with explanatory error |

User can override via the `quantization` dropdown: `auto | fp16 | bf16 | 8bit | 4bit`.

If a load OOMs → automatically retry one step down (bf16 → 8bit → 4bit) and surface a warning in the status; fail hard if 4bit also OOMs.

### 5.4 LLMAdapter extension (`src/apps/comic_gen/llm_adapter.py`)

Add a third branch:

```python
def __init__(self):
    self.provider = os.getenv("LLM_PROVIDER", "dashscope").lower()
    ...

def chat(self, messages, model=None, response_format=None) -> str:
    if self.provider == "local":
        from src.llm_local.manager import get_manager
        # response_format ignored (no native JSON mode); rely on prompts
        return get_manager().chat_sync(messages)   # blocks via asyncio.run_coroutine_threadsafe
    # existing dashscope / openai branches unchanged
```

Note: ScriptProcessor calls `LLMAdapter.chat()` from request handlers (already running in async context via FastAPI). The adapter exposes `chat()` as sync (existing contract). For local provider, we bridge sync → async via `asyncio.run_coroutine_threadsafe` against the running loop. Implementation detail in plan.

### 5.5 LocalLLMPanel (frontend)

```
┌─ Local LLM ────────────────────────────────────┐
│  HF Model ID                                    │
│  [Qwen/Qwen3-8B-Instruct                    ]  │
│  Quantization: [Auto ▾]                         │
│  Idle unload after: [3] seconds                 │
│  [Save Configuration]                           │
│                                                 │
│  ─────────────────────────────────────────────  │
│  Status: ● READY                                │
│  Model: Qwen/Qwen3-8B-Instruct                 │
│  VRAM: 15.2 / 24.0 GB                          │
│  Last used: 2 seconds ago                       │
│  [Load Now]  [Unload]  [Test]                   │
│                                                 │
│  ─────────────────────────────────────────────  │
│  Cached Models (output/models/LLM/)             │
│  ▸ Qwen/Qwen3-8B-Instruct      15.4 GB  [🗑]   │
│  ▸ Qwen/Qwen3-4B-Instruct       7.8 GB  [🗑]   │
└────────────────────────────────────────────────┘
```

State badge colors: gray UNLOADED, blue DOWNLOADING/LOADING, green READY, red ERROR.

The whole panel only renders when LLM Provider = `Local` is selected in the parent SettingsPage.

## 6. API Contracts

### `POST /llm/local/configure`

```json
// Request
{
  "hf_id": "Qwen/Qwen3-8B-Instruct",
  "quant": "auto",          // auto | fp16 | bf16 | 8bit | 4bit
  "idle_seconds": 3
}
// Response: 200 OK
{ "ok": true, "persisted": true }
```

Side effects:
- Persists `LLM_PROVIDER=local`, `LOCAL_LLM_HF_ID`, `LOCAL_LLM_QUANT`, `LOCAL_LLM_IDLE_SEC` to `.env` via existing `set_key`.
- If currently loaded model id differs → unload immediately.
- Does NOT auto-load. Loading is lazy or via explicit `/load`.

### `POST /llm/local/load`

Forces immediate load. Returns when state is READY or ERROR.

```json
// Response success
{ "state": "READY", "vram_used_mb": 15580, "vram_total_mb": 24576, "elapsed_sec": 47.2 }
// Response error
{ "state": "ERROR", "error": "OOM at bf16, retried 8bit also OOM" }
```

### `POST /llm/local/unload`

Idempotent. Always returns `{ "state": "UNLOADED" }`.

### `GET /llm/local/status`

```json
{
  "state": "READY",
  "hf_id": "Qwen/Qwen3-8B-Instruct",
  "quant_mode": "bf16",
  "vram_used_mb": 15580,
  "vram_total_mb": 24576,
  "last_used_ts": 1712345678.9,
  "idle_seconds": 3,
  "error": null
}
```

Frontend polls this every 2 seconds while panel is visible.

### `POST /llm/local/test`

Runs a hardcoded mini chat (`[{"role": "user", "content": "Say hello in one word."}]`) and returns the raw response. Test passes if the response is non-empty (no strict equality on content — different models phrase differently).

```json
{ "ok": true, "elapsed_sec": 1.4, "response": "Hello" }
```

Purpose: confirm tokenizer + chat_template + generate + decode end-to-end with the currently configured model.

### `GET /llm/local/cached`

Lists `output/models/LLM/` cache directory contents, parsed from HF cache layout (`models--<org>--<name>` snapshots):

```json
[
  { "hf_id": "Qwen/Qwen3-8B-Instruct", "size_bytes": 16544000000, "snapshot_path": "..." },
  { "hf_id": "Qwen/Qwen3-4B-Instruct", "size_bytes": 8400000000, "snapshot_path": "..." }
]
```

### `DELETE /llm/local/cached?hf_id=...`

Removes that model's snapshot directory. Refuses if currently loaded (must `/unload` first).

## 7. Configuration & Persistence

`.env` keys (added):

```
LLM_PROVIDER=local                       # existing key, new value
LOCAL_LLM_HF_ID=Qwen/Qwen3-8B-Instruct
LOCAL_LLM_QUANT=auto
LOCAL_LLM_IDLE_SEC=3
```

`HF_HOME` is set programmatically at app startup (before any `transformers` import path matters) to `<project_root>/output/models/LLM/`. This redirects all HF caches into the project tree.

Default values shipped:
- `LOCAL_LLM_QUANT=auto`
- `LOCAL_LLM_IDLE_SEC=3`
- `LOCAL_LLM_HF_ID` unset (user must configure before first load)

## 8. Lifecycle Examples

**Cold first call:**

```
t=0    Frontend → POST /script/extract
t=0    ScriptProcessor.extract() → LLMAdapter.chat()
t=0    Adapter sees provider=local → ModelManager.chat()
t=0    State UNLOADED → trigger load (acquire lock)
t=0    Download Qwen3-8B from HF (~16GB, depends on bandwidth)
t=300  Load weights to GPU (bf16, ~30s)
t=330  State READY, run apply_chat_template + generate
t=345  Return response, last_used_ts=345
t=348  Idle watcher: 345+3 ≤ 348 → unload
t=349  State UNLOADED, VRAM freed
```

**Hot pipeline (sequential calls within 3s):**

```
t=0    chat #1 → load + run → READY, last_used=10
t=11   chat #2 (within 3s of last_used) → use loaded model → last_used=14
t=15   chat #3 → use loaded model → last_used=18
t=21   idle watcher: 18+3 < 21 → unload
```

**Two-model swap:**

```
State: READY (Qwen3-8B)
User saves new config: hf_id=Qwen/Qwen3-14B
configure() → unload Qwen3-8B → state UNLOADED
Next chat() → lazy-load Qwen3-14B
```

## 9. Error Handling

| Scenario | Handling |
|----------|----------|
| HF download network failure | State → ERROR with message; chat() raises `RuntimeError`; surfaces to existing FastAPI error response. Next call retries from scratch. |
| HF id not a causal LM (e.g., embedding model) | Caught at `AutoModelForCausalLM.from_pretrained` → ERROR with explanatory message. |
| OOM during load | Auto step down quant (bf16→8bit→4bit). If 4bit also OOM → ERROR. |
| OOM during inference | ERROR; recommend smaller `max_new_tokens` or smaller model. |
| Unload while chat in flight | Lock prevents this — unload waits. |
| User deletes cached model that's currently loaded | DELETE returns 409 Conflict. |
| `LOCAL_LLM_HF_ID` empty when chat called | ERROR with message: "Configure local LLM first". |
| GPU not detected | ERROR at startup with clear message ("CUDA not available; local LLM provider requires GPU"). |

## 10. Dependencies (added to `requirements.txt`)

```
torch>=2.3.0,<3.0.0                # CUDA 12.x build via PyTorch index
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.43.0               # Native Windows support since 0.43
huggingface_hub>=0.25.0
sentencepiece>=0.2.0               # Tokenizer dependency for some Qwen variants
```

**Install size impact:** ~5–7 GB once installed (mostly torch + CUDA wheels). One-time.

`bitsandbytes` on Windows requires the >=0.43 native wheel. We pin `>=0.43.0`.

## 11. Risks & Open Items

**Risks (acknowledged):**

| Risk | Mitigation |
|------|------------|
| Cold start 30–90s every time idle expires | User accepted (option A). UI shows clear LOADING state. |
| HF download speed without mirror | User chose international endpoint. Add docs note about how to set `HF_ENDPOINT` later if needed. |
| transformers `device_map="auto"` may split layers across CPU/GPU on insufficient VRAM, silently slow. | Detect VRAM up front, refuse to load if model size > available; force user to pick smaller model or quant. |
| `bitsandbytes` Windows wheel install hiccups | Document Python 3.12 requirement (already met by current venv). |
| HF format models without chat_template (rare for instruct models) | Detect missing template at configure time, show error. |
| Sync→async bridge (LLMAdapter is sync, ModelManager is async) | Use `asyncio.run_coroutine_threadsafe` against the FastAPI loop captured at startup. Implementation detail. |
| Pipeline calls `response_format={"type":"json_object"}` — local mode silently ignores | Existing prompts already say "Return STRICTLY JSON". Acceptable for v1. Add JSON-retry-on-parse-fail in `llm.py` if it becomes a problem (already partially exists). |

**Open items (deferred to plan stage, not blocking):**

- Whether to wire HF download progress into `/llm/local/status` for UI progress bar (nice-to-have; can add later)
- Whether `/llm/local/cached` returns models still being downloaded (mid-download)
- Where to surface per-call timing metrics (currently nowhere; add to logs only)

## 12. Future Phases (informational only)

The `output/models/{LLM,Image,Video}/` layout is intentional. Later:
- **Image models** (e.g., Flux, SDXL via diffusers) → same pattern: `ImageModelManager`, `output/models/Image/`, `IMAGE_MODEL_PROVIDER=local`.
- **Video models** (e.g., Wan2.1 via diffusers) → same pattern: `output/models/Video/`.

Each will have its own manager with its own state, but the overall architecture (configure → lazy load → 3s idle unload → cached list UI) is identical and can be reused.

---

## Appendix A — Files NOT touched (intentional)

- `scripts/start-backend.js`, `scripts/dev-setup.js`, `dev.bat` — no changes; new behavior is additive.
- `src/utils/oss_utils.py` — unrelated.
- Existing `dashscope` / `openai` provider paths — completely untouched, fully backward compatible.
