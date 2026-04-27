# Download Status (downloading / paused / complete) — Design

**Date:** 2026-04-26
**Status:** Approved (brainstorming, inline)

---

## Goal

`/llm/local/cached` currently reports `size_bytes` but doesn't distinguish:
1. **downloading** — partial blob exists, worker actively writing
2. **paused** — partial blob exists, no worker writing (e.g., backend was restarted, or load was never re-triggered)
3. **complete** — fully downloaded, ready to load into VRAM

Without this distinction, the UI shows partial downloads identically to complete ones, and there's no "resume" affordance after a restart.

## Non-Goals

- Not adding a Pause button (user can just close browser; no in-flight pause primitive — the partial stays naturally)
- Not adding download speed / ETA (separate concern)
- Not changing the cancel-on-error or cancel-on-explicit-✕ semantics
- Not implementing a fully accurate progress bar against expected total size (best-effort: query HfApi when available, else show partial only)

## State detection

For each cached repo `models--<org>--<name>`:

```
def download_status(snapshots_dir, blobs_dir, manager_state, manager_hf_id):
    has_complete_files = (
        snapshots_dir.exists()
        and any(
            f.is_file() and not f.name.endswith(".incomplete")
            for snap in snapshots_dir.iterdir() if snap.is_dir()
            for f in snap.iterdir()
        )
    )
    if has_complete_files:
        return "complete"

    if manager_state == "LOADING" and manager_hf_id == hf_id:
        return "downloading"
    return "paused"
```

## Optional: expected total size

Use `HfApi().get_paths_info(repo_id, paths=[gguf_filename])` to get the file's `size`. Cache the result in-memory by `(hf_id, gguf_filename)` to avoid hammering HF API on every /cached call (which is polled every 1.5–3s).

If HF API call fails (no network, repo gone, etc.), `expected_size_bytes = None` — UI just doesn't show the percentage.

## API contract changes

`CachedModelInfo` adds two optional fields:

```python
class CachedModelInfo(BaseModel):
    hf_id: str
    size_bytes: int
    snapshot_path: str
    gguf_files: Optional[List[str]] = None
    active_gguf_file: Optional[str] = None
    download_status: Literal["downloading", "paused", "complete"] = "complete"
    expected_size_bytes: Optional[int] = None
```

`download_status` defaults to "complete" so old clients (and entries that bypass the new logic) keep behaving sensibly.

## Frontend display

```
┌────────────────────────────────────────┐
│  LOCAL  GGUF Q4_K_M       ✓             │
│  Qwen3-30B-A3B-Instruct-2507-GGUF      │
│  17.28 GB                               │
└────────────────────────────────────────┘  ← complete (existing)

┌────────────────────────────────────────┐
│  LOCAL  GGUF Q4_K_M  🔵 DOWNLOADING ⌛   │
│  Qwen3.6-35B-A3B-APEX-GGUF             │
│  ↓ 2.6 / 14.3 GB (18%)        [✕]      │
└────────────────────────────────────────┘  ← downloading (active)

┌────────────────────────────────────────┐
│  LOCAL  GGUF Q4_K_M  🟡 PAUSED          │
│  Qwen3.6-35B-A3B-APEX-GGUF             │
│  ⏸ 2.6 / 14.3 GB (18%) [▶ Resume] [✕]  │
└────────────────────────────────────────┘  ← paused (resumable)
```

**Resume button** = `POST /llm/local/configure` (with current hf_id + gguf_file from the cached entry) + `POST /llm/local/load` (which fires off, doesn't await). hf-xet protocol resumes from the .incomplete blob automatically.

## Tests

Backend (4 new):
- `download_status == "complete"` when snapshot has non-incomplete files
- `download_status == "downloading"` when blob `.incomplete` exists + manager LOADING + hf_id matches
- `download_status == "paused"` when blob `.incomplete` exists + manager not LOADING (or different hf_id)
- `download_status == "paused"` when blob exists but snapshot dir is empty AND manager isn't loading this id

Frontend: visual only (no automated test); manual e2e to verify states render correctly.

## Files

Modified:
- `src/llm_local/api.py` (+ ~50 lines for status detection + HfApi call)
- `frontend/src/lib/api.ts` (+ 2 fields on `CachedModelInfo`)
- `frontend/src/components/common/LLMModelSection.tsx` (+ badge rendering, Resume button, ~40 lines)
- `tests/llm_local/test_api.py` (+ 4 new tests)

No new files. No backend route additions.

## Risks

| Risk | Mitigation |
|------|------------|
| HfApi.get_paths_info call adds latency to /cached | In-memory cache `{(hf_id, file): size}`; the call only runs once per session per file |
| HfApi anonymous calls have rate limits too | If it fails, fall back to `expected_size_bytes = None`; UI just shows partial without total |
| Status detection mid-rename race (file is being moved from blobs to snapshots when we scan) | Both states resolve to "complete" anyway; transient false reads are fine |
| User's currently-downloading 2.6GB should not be disturbed by reload | Only adds READ-ONLY logic to /cached + frontend rendering; no destructive ops introduced |
