# Embedded llama-server Runtime — Design

**Date:** 2026-04-26
**Status:** Pending user approval
**Scope:** Replace the in-process `GGUFRuntime` (llama-cpp-python) with a subprocess `EmbeddedServerRuntime` that spawns upstream llama.cpp's `llama-server.exe`. Eliminates LM Studio dependency. Inherits and extends ModelManager's existing state machine.

---

## Goal

Run modern GGUFs (including `qwen35moe`, future architectures) inside lumenx without:
- Depending on LM Studio (no GUI install required)
- Building llama-cpp-python from source on Windows (MSVC UTF-8 hell)
- Being stuck on llama-cpp-python 0.3.4's old llama.cpp commit

Done by:
- Auto-downloading upstream llama.cpp's prebuilt Windows CUDA binaries from GitHub releases (one-time, ~200MB)
- Spawning `llama-server.exe -m <gguf> --port 17178` as a subprocess
- Using lumenx's existing `openai` HTTP client to talk to `http://127.0.0.1:17178/v1`
- ModelManager state machine (LOADING/READY/ERROR/idle-unload) drives subprocess lifecycle

## Non-Goals

- Not bundling the binary in git (200MB; license headers complex)
- Not supporting Linux/macOS in this iteration (Windows CUDA only — matches user env)
- Not exposing model-loading hyperparameters in UI (--n-gpu-layers, --ctx-size etc. — use llama.cpp defaults; configurable via env later)
- Not implementing streaming responses (existing `LLMAdapter.chat()` returns full string)
- Not removing llama-cpp-python; it's already installed but we just stop using it. Removal is a follow-up cleanup commit.
- Not running multiple GGUF models concurrently (single-GPU constraint; existing single-runtime semantics preserved)

## Architecture

```
ScriptProcessor (existing)
    │ self.llm.chat(messages)
    ▼
LLMAdapter (existing, untouched)
    provider == "local" → ModelManager.chat_sync()
    │
    ▼
ModelManager (existing state machine, factory unchanged at the top level)
    │ _default_runtime_factory(hf_id, quant, gguf_file)
    │     is_gguf_repo(hf_id) → EmbeddedServerRuntime  (NEW)
    │     else                → LocalRuntime           (existing, transformers)
    ▼
EmbeddedServerRuntime (NEW)
    .load():
      1. ensure_binary_installed()       # downloads llama.cpp release on first use
      2. ensure_gguf_downloaded()         # uses huggingface_hub.hf_hub_download (existing)
      3. spawn llama-server.exe -m <gguf-path> --port 17178 ...
      4. poll http://127.0.0.1:17178/health until 200 OK (or timeout)
    .chat(messages):
      1. POST localhost:17178/v1/chat/completions  (using openai client)
      2. parse + return content
    .unload():
      1. terminate subprocess (graceful → kill)
      2. wait for exit (frees VRAM via OS process cleanup)
```

Subprocess advantage over in-process llama-cpp-python:
- VRAM is freed reliably on process exit (no CUDA context leak)
- Crashes (segfault, OOM) don't take down lumenx backend
- Easy upgrade: swap binary, no Python ABI alignment

## File Layout

**New:**
```
src/llm_local/
├── runtime_server.py          # EmbeddedServerRuntime: subprocess wrapper
├── llama_cpp_release.py       # download + extract llama.cpp release zip
└── server_proc.py             # process supervision: spawn, health-check, terminate

tests/llm_local/
├── test_runtime_server.py     # subprocess lifecycle, mocked subprocess + httpx
├── test_llama_cpp_release.py  # download/extract logic (mocked HTTP)
```

**Modified:**
```
src/llm_local/manager.py       # factory dispatch swap: GGUFRuntime → EmbeddedServerRuntime
src/llm_local/api.py           # GET /llm/local/runtime → reports binary install status
requirements.txt               # remove llama-cpp-python (not used anymore); add httpx (already transitively)
```

**Deleted (not in this PR; deferred cleanup):**
```
src/llm_local/runtime_gguf.py  # superseded; can be removed in a follow-up commit
tests/llm_local/test_runtime_gguf.py
```

## Components

### `llama_cpp_release.py` — auto-update binary manager

**No version pinning.** On every backend startup, check GitHub for latest release; if newer than installed, download in background.

```python
# Constants — only the URL pattern is fixed; tag is fetched live from GitHub
ASSET_NAME = "cudart-llama-bin-win-cuda-13.1-x64.zip"
GITHUB_LATEST_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
RELEASE_URL_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/" + ASSET_NAME
)
RUNTIME_PARENT = PROJECT_ROOT / "output" / "runtime"


class LlamaCppManager:
    """Singleton managing the installed llama.cpp release(s).

    Layout: output/runtime/llama.cpp-<tag>/llama-server.exe
    Multiple versions can coexist; current_exe_path() returns the
    lexicographically-largest installed tag (i.e. newest by build number).
    """

    def installed_versions(self) -> list[str]:
        """List 'b8941', 'b9001' etc. dirs present under RUNTIME_PARENT."""

    def current_version(self) -> Optional[str]:
        """Highest-numbered installed version, or None if nothing installed."""

    def current_exe_path(self) -> Optional[Path]:
        """Path to llama-server.exe of the current_version()."""

    def latest_remote(self, timeout: float = 10.0) -> Optional[str]:
        """GitHub API call to get latest tag. Returns None on network failure."""

    async def ensure_latest_available(self) -> str:
        """Background task: fetch latest tag, compare with installed, download
        if newer. Returns the version that ended up being current. Never raises;
        on failure, logs and returns whatever was already installed (or None).
        Safe to call from app startup."""

    def cleanup_old(self, keep: int = 2) -> list[str]:
        """Delete all but the `keep` most recent installed versions to bound
        disk usage. Returns list of deleted version tags."""
```

**Triggering**: lumenx backend's `@app.on_event("startup")` calls
`asyncio.create_task(LlamaCppManager().ensure_latest_available())`. Fire-and-forget;
doesn't block startup. While download is in progress, `current_exe_path()` keeps
returning the previously-installed version (so any chat request still works).

**Concurrency safety**: a `.lock` file in RUNTIME_PARENT during downloads. If
backend restarts mid-download, we resume on next start; if the lock is stale
(>10 min) we treat the install dir as corrupt → re-download.

### Behavior matrix

| Local state | GitHub reachable | Latest > local | Outcome |
|------------|------------------|---------------|---------|
| nothing installed | yes | n/a | download latest, block until done if user requests load before |
| nothing installed | no | n/a | next model load → ERROR "can't reach github, no local install" |
| have b8941 | yes | yes (b9000 latest) | b9000 downloads in background; current requests use b8941; new server spawns use b9000 once download finishes |
| have b8941 | yes | no (b8941 is latest) | no-op |
| have b8941 | no | n/a | warn in log, use b8941 |
| download in progress | (irrelevant) | (irrelevant) | next call to ensure_latest_available is no-op (lock held) |

### Cleanup

After a successful new install, `cleanup_old(keep=2)` removes all but the 2 most recent installed versions. Default 2 (current + previous) gives a one-step rollback safety. Never deletes the version of the currently-running server (separate check via process tracking).

### `server_proc.py` — subprocess primitive

```python
class LlamaServerProcess:
    def __init__(self, exe: Path, gguf_path: Path, port: int = 17178,
                 n_gpu_layers: int = -1, n_ctx: int = 8192):
        ...
    def start(self, startup_timeout: float = 60.0) -> None:
        """Spawn process; block until /health returns 200 OK or raise."""
    def is_alive(self) -> bool:
        ...
    def terminate(self, grace_seconds: float = 5.0) -> None:
        """SIGTERM, wait, then SIGKILL if needed. Frees VRAM via OS cleanup."""
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
```

Owns the `subprocess.Popen` object. Provides health-check polling. Handles Windows process termination cleanly (no `taskkill` shenanigans — uses `Popen.terminate()` then `kill()`).

### `runtime_server.py` — drop-in replacement for GGUFRuntime

Same interface as `LocalRuntime` (load/chat/unload + `LoadResult`). Internally:
- `load()` calls `ensure_binary_installed()` → `hf_hub_download(gguf)` → starts `LlamaServerProcess`
- `chat()` uses `openai.OpenAI` client pointed at `http://127.0.0.1:17178/v1`
- `unload()` calls `proc.terminate()`

### Manager dispatch — minimal change

```python
def _default_runtime_factory(hf_id, quant, gguf_file=None):
    if is_gguf_repo(hf_id):
        from .runtime_server import EmbeddedServerRuntime
        return EmbeddedServerRuntime(hf_id=hf_id, gguf_file=gguf_file)
    return LocalRuntime(hf_id=hf_id, quant=quant)
```

That's the entire ModelManager change. State machine, idle watcher, cancel logic — all reused as-is. Subprocess termination from cancel/idle just calls `runtime.unload()` which calls `proc.terminate()`.

### `/llm/local/runtime` — new diagnostic endpoint

```json
GET /llm/local/runtime
{
  "version": "b6797",
  "installed": true,
  "exe_path": ".../output/runtime/llama.cpp-b6797/llama-server.exe",
  "size_bytes": 209715200
}
```

UI uses this to show "llama.cpp runtime ready" or "Click to install (200MB)". Not strictly required for v1 but nice to have for visibility.

## Data Flow Examples

### Backend startup (first ever)

```
t=0    Backend boots
t=0    @app.on_event("startup") fires:
         - existing local-llm hooks
         - asyncio.create_task(LlamaCppManager().ensure_latest_available())
t=0.1  Startup completes; backend serves requests immediately
t=2    Background task: GET github API → tag b8941
t=2    No local install → download cudart-llama-bin-win-cuda-13.1-x64.zip (~374MB)
t=60   Extract → output/runtime/llama.cpp-b8941/
t=60   current_exe_path() now returns the path (was None before)
```

### Backend startup (already have b8941, latest is b9000)

```
t=0    Backend boots, current_exe_path() = .../llama.cpp-b8941/llama-server.exe
t=2    Background task: GitHub says b9000 → newer → download in background
t=60   Extract llama.cpp-b9000/
t=60   current_exe_path() now returns b9000
t=60   cleanup_old(keep=2) — both b8941 and b9000 stay (within keep limit)
```

### First time loading anthfu (binary still being downloaded)

```
t=0    User clicks Load anthfu
t=0    runtime.load() — current_exe_path() is None (binary not yet ready)
t=0    raise RuntimeError("llama.cpp is being downloaded — try again in ~1 minute")
       state ERROR with "llama.cpp downloading…" message
t=60   Download completes
t=60   User retries (or auto-retry — TBD if we add retry logic) → succeeds
```

### Steady-state load (binary installed, latest)

```
t=0    runtime.load():
       - exe ready, GGUF cached → skip downloads
       - spawn server, poll /health
t=15s  state READY
```

### A new release lands while server is running

```
state READY (b8941 spawned)
@app.on_event("startup") long since fired; ensure_latest_available was no-op then
NO automatic mid-session upgrade. (Avoid disturbing in-flight chats.)
Next backend restart will pick up the newer release.
```

### Auto-unload after idle

```
state READY, idle watcher ticks every 1s
t=N    last_used > idle_seconds (3s default)
       runtime.unload() → proc.terminate() → SIGTERM
       process exits within 5s, VRAM freed by OS
       state UNLOADED
```

### Cancel while loading

```
ModelManager.cancel() injects CancelledByUser into worker thread
   ↓ (worker is in /health poll loop or inside subprocess.Popen.wait)
   Exception bubbles up, EmbeddedServerRuntime.load() catches it
   Calls proc.terminate() to clean up half-started server
   state UNLOADED, no zombie subprocess
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Binary download fails (network) | state ERROR with explanatory message; user can retry |
| Binary zip corrupt | extract fails → ERROR; on next /load retry, we delete partial dir + redownload |
| `llama-server.exe` exits during /health poll (e.g. CUDA not found) | health-check timeout → state ERROR with stderr tail |
| Port 17178 already in use | startup fails fast (server can't bind) → ERROR with clear message |
| Subprocess hangs on terminate | After 5s grace, force-kill via `Popen.kill()` |
| Multiple concurrent /load calls | Existing ModelManager lock serialises (no change needed) |
| GGUF architecture not supported by THIS llama.cpp build | server reports load error in stderr → health stays unhealthy → ERROR with stderr |

## Tests

**Unit (mock subprocess + httpx):**
1. `test_llama_cpp_release.py`:
   - `is_installed()` true/false based on file existence
   - `download_and_extract()` writes llama-server.exe to RUNTIME_DIR (mocked urlopen + ZipFile)
2. `test_server_proc.py`:
   - `start()` polls /health, transitions to ready
   - `start()` raises on startup_timeout
   - `terminate()` sends SIGTERM then SIGKILL on grace timeout
3. `test_runtime_server.py`:
   - `load()` installs binary if missing, downloads gguf, starts proc, returns LoadResult
   - `chat()` POSTs to right URL, returns content
   - `unload()` is idempotent

**Integration (no network for CI; requires user to run manually):**
- `test_runtime_server_integration.py`: actually spawn llama-server with a tiny GGUF (Qwen2.5-0.5B-Instruct-GGUF), do one chat. Marked `@pytest.mark.slow`, skipped without GPU.

**Existing tests:** `test_manager.py` factory dispatch tests need updating — replace assertions about GGUFRuntime with EmbeddedServerRuntime references.

## Risks

| Risk | Mitigation |
|------|------------|
| GitHub release URL pattern changes | Asset name (`cudart-llama-bin-win-cuda-13.1-x64.zip`) and URL template are constants in code; if a future release renames assets, ensure_latest_available logs an error and falls back to currently-installed version. User can manually drop binary in `output/runtime/llama.cpp-<tag>/` and lumenx auto-detects. |
| GitHub API rate limit (anonymous: 60 req/h per IP) | We call /releases/latest at most once per backend startup. With normal restart cadence, never hit limit. If hit (HTTP 403), fall back to local. |
| Network slow/down at startup | ensure_latest_available is fire-and-forget; doesn't block startup. Local install (if present) keeps working. |
| User starts backend offline, then comes online | Latest check only fires at startup. Either restart, or expose a /llm/local/runtime/check endpoint to manually trigger (deferred). |
| Bumping to a new release breaks old GGUFs | New version of llama.cpp may rarely deprecate older quant formats. cleanup_old(keep=2) leaves the previous version on disk so a manual rollback is possible (rename the dir or edit a fallback constant). |
| 200MB download too slow / unreliable for some users | Document manual install path: drop `llama-server.exe` + DLLs into `output/runtime/llama.cpp-bXXXX/` and lumenx auto-detects |
| Port 17178 conflicts with something else | Configurable via env var `LLAMA_SERVER_PORT`; default 17178 (lumenx itself is 17177, naturally adjacent) |
| Latest llama.cpp build STILL doesn't support qwen35moe | Easy verification step before committing: `strings llama-server.exe | grep qwen35moe`. If absent, pin a different commit. |
| User has no CUDA toolkit (only torch's bundled) | llama.cpp Windows CUDA binary uses statically-linked CUDA runtime — should work without separate toolkit. We test before claiming. |
| llama-cpp-python still installed but unused | Acceptable for v1; remove in follow-up cleanup commit. Keeps PR scope tight. |

## Verification Done (2026-04-26)

1. **Release URL pattern verified** via `https://api.github.com/repos/ggerganov/llama.cpp/releases/latest`:
   - Latest tag: `b8941`
   - Asset: `cudart-llama-bin-win-cuda-13.1-x64.zip` (374 MB) — bundles CUDA runtime
   - Org renamed `ggerganov` → `ggml-org` (URL auto-redirects)
2. **qwen35moe support**: LM Studio v2.14.0 (older llama.cpp commit, verified directly via `strings llama.dll`) already has `qwen35moe`. b8941 is many weeks newer → essentially certain.
3. **Defensive check on first install**: code grep `strings llama-server.exe` for `qwen35moe`; if absent, log warning at install time so user knows to use a different model.
