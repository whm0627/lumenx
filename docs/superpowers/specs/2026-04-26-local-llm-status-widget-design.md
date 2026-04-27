# Local LLM Status Widget — Design

**Date:** 2026-04-26
**Status:** Approved (brainstorming)
**Scope:** Frontend-only. No backend changes.
**Reference (prior specs):**
- [2026-04-25-local-llm-runtime-design.md](./2026-04-25-local-llm-runtime-design.md)
- [2026-04-25-llm-picker-and-gguf-runtime-design.md](./2026-04-25-llm-picker-and-gguf-runtime-design.md)

---

## 1. Goal

A compact, always-visible status widget in the left `GlobalSidebar` footer area that reflects the active local LLM's lifecycle: idle, downloading (with on-disk size growing), ready (with VRAM usage), or error. Lets the user see at a glance whether the local model is loaded / loading / waiting, without opening any modal.

## 2. Non-Goals

- Cloud provider status (DashScope / OpenAI) — not shown
- Per-call inference progress (token streaming, etc.)
- Cancel-download button (would require backend support; defer)
- Download percentage with total — option B in brainstorming was rejected for v1; we show on-disk size growing without a denominator
- Top bar / bottom-right floating positions — chose sidebar footer per brainstorm
- VRAM tracking parity for GGUF runtime — GGUF reports `vram_used_mb=0`; we display "VRAM —" or skip the line in that case

## 3. Architecture

```
GlobalSidebar.tsx
├─ Branding (existing)
├─ Nav (existing)
├─ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
└─ Footer (existing)
    └─ LocalLLMStatusWidget (NEW, conditionally rendered)
        - polls /llm/local/status every 2s
        - polls /llm/local/cached every 2s (only when LOADING, to read on-disk growth)
        - reads /config/env once on mount to know LLM_PROVIDER
        - emits a custom event 'open-llm-settings' on click
            → page.tsx listens, switches GlobalTab to 'settings'
```

The widget is **self-contained**: parent `GlobalSidebar` only renders `<LocalLLMStatusWidget />` — no props, no upward state.

## 4. File Layout

**New:**
```
frontend/src/components/layout/LocalLLMStatusWidget.tsx   (~140 lines)
```

**Modified:**
```
frontend/src/components/layout/GlobalSidebar.tsx          (footer block: insert widget above v0.1.0)
frontend/src/app/page.tsx                                 (add 'open-llm-settings' event listener that sets activeTab='settings')
```

## 5. Component Behaviour

### Render conditions

```typescript
if (env?.LLM_PROVIDER !== "local") return null;
if (!env?.LOCAL_LLM_HF_ID) return null;
```

(Cloud users see no widget at all — keeps the sidebar clean.)

### Data fetched

- `GET /config/env` once on mount: `LLM_PROVIDER`, `LOCAL_LLM_HF_ID`, `LOCAL_LLM_GGUF_FILE`
- `GET /llm/local/status` every 2s (always, while widget is mounted)
- `GET /llm/local/cached` every 2s **only while state ∈ {LOADING, DOWNLOADING}** (to read partial download size)

### State → display

| state | dot color | line 1 (state) | line 3 (detail) |
|-------|-----------|----------------|-----------------|
| `UNLOADED` | gray (`bg-gray-500`) | `IDLE` | `"<n>s ago"` (relative time of last_used_ts), `—` if never used |
| `LOADING` / `DOWNLOADING` | blue (`bg-blue-500 animate-pulse`) | `LOADING` | `"⬇ <formatted size>"` from cached endpoint, falls back to `"…"` if not yet on disk |
| `READY` | green (`bg-green-500`) | `READY` | `"VRAM <X.X GB>"` if `vram_used_mb > 0`, else `"—"` (GGUF case) |
| `ERROR` | red (`bg-red-500`) | `ERROR` | error message (truncated to ~40 chars), full text in `title=` tooltip |

Line 2 always shows the model: `shortHfId(LOCAL_LLM_HF_ID)` (org prefix stripped). For GGUF models, append a small badge `GGUF Q4_K_M` next to it.

### Click

```typescript
onClick={() => window.dispatchEvent(new CustomEvent('open-llm-settings'))}
```

`page.tsx` listens for this event and:
```typescript
useEffect(() => {
  const handler = () => setActiveTab('settings');
  window.addEventListener('open-llm-settings', handler);
  return () => window.removeEventListener('open-llm-settings', handler);
}, []);
```

(Custom event because the widget lives deep inside `GlobalSidebar` which already takes `onTabChange` as a prop, but we don't want to thread a special action through the sidebar API.)

## 6. Visual Design

```
┌────────────────────────────┐
│ ● LOADING                   │  text-xs, font-mono on the state word
│ Qwen3.6-35B-A3B-APEX-GGUF  │  text-sm, white, truncate
│ ⬇ 1.2 GB                    │  text-xs, gray-400
└────────────────────────────┘
       v0.1.0                      (existing footer line, kept)
```

Total height ~60px. Padding consistent with existing nav items (px-4 py-3).

Hover: subtle highlight (`hover:bg-white/5`), cursor-pointer. Tooltip (`title=`) shows full hf_id + gguf_file + last_used absolute timestamp.

### Format helpers (reused from LLMModelSection.tsx)

- `shortHfId("anthfu/Qwen3.6-35B-A3B-APEX-GGUF")` → `"Qwen3.6-35B-A3B-APEX-GGUF"` (truncated by CSS `truncate` if container width exceeded)
- `shortGgufFile("model-Q4_K_M.gguf")` → `"GGUF Q4_K_M"` (rendered as a small badge)
- `formatBytes(1234567890)` → `"1.15 GB"`
- `formatRelativeTime(ts)` → `"23s ago"` / `"5m ago"` / `"never"`

These already exist in `LLMModelSection.tsx`. To avoid duplication: extract into `frontend/src/lib/formatLLM.ts` and import from both.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Polling /cached every 2s during LOADING is wasteful for the rest of the app | Only poll while state ∈ LOADING/DOWNLOADING; resume normal status-only polling on READY/UNLOADED |
| Sidebar width (56px) may truncate the model name awkwardly | CSS `truncate` + tooltip with full id; acceptable |
| Custom event coupling to page.tsx feels hacky | It's a small surface (one event); alternative was threading callback through GlobalSidebar API which is worse |
| GGUF runtime reports vram_used_mb=0; widget would show "VRAM 0 GB" misleadingly | Render "—" instead of "0.0 GB" when value is 0 in READY state |
| If user navigates between pages, polling restarts each mount | Acceptable — widget lives in sidebar which is in AppShell, so it doesn't unmount on tab switch |

## 8. Out-of-Scope Future

- Cancel-download button (needs backend abort signal)
- Token-streaming chat indicator
- Multiple models pinned status
- VRAM polling for GGUF via nvidia-ml-py
