# Local Video (Wan2.2-S2V-14B via GGUF) — design

**Date:** 2026-04-28
**Status:** approved (pending user review)
**Roadmap position:** lumenxDev local-stack #5 (after LLM, T2I, I2I, TTS)
**Outcome:** `VIDEO_PROVIDER=local` produces audio-driven lipsync video clips on a single RTX 4090 24GB, comfortably under the VRAM ceiling, without ComfyUI as a dependency.

## Background

The previous validation against Wan official `generate.py` confirmed the model loads (1m27s, 23.8 GB peak VRAM, 100% GPU compute) but is on the edge of OOM on a 24GB card and slow per step. The community-standard fast path on consumer GPUs is **GGUF quantization** (city96/ComfyUI-GGUF) plus optionally **Sage Attention** — `Wan2.2-S2V-14B Q4_K_S` is 13 GB on disk vs 28 GB fp16. Reported 5s @ 1024×574 in ~7 minutes on 4090 with Q8 + Sage.

Both pieces live inside ComfyUI today. lumenxDev runs a custom Python backend (no ComfyUI) and integrates local models via in-process modules (`img_local`, `audio_local`). This spec ports the GGUF + Sage path standalone, mirroring those modules.

## Goals

1. Audio-driven lipsync video generation (same Wan2.2-S2V-14B model, same official inference path) at meaningfully lower VRAM and meaningfully higher speed than the un-quantized baseline.
2. Three quantization tiers: `fp16` (no quant), `Q8_0`, `Q4_K_S` — selectable at runtime via env / settings UI.
3. Sage Attention 2.x as opt-in speedup, not a hard dependency (graceful fallback if user can't install the wheel).
4. Generic GGUF infra that doesn't lock to the S2V variant — adding I2V-A14B / Animate-14B later means writing a new model wrapper, not a new dequant kernel.

## Non-goals

- Other Wan2.2 variants (T2V-A14B, I2V-A14B, Animate-14B, TI2V-5B). Architecture supports them; v1 only ships S2V-14B.
- Quantization tiers beyond {fp16, Q8_0, Q4_K_S}. The full ComfyUI-GGUF spread (Q2_K…Q6_K) is YAGNI for the 4090 sweet spot.
- ComfyUI workflow compatibility / interop.
- Runtime quant switching without an unload + reload (matches `img_local` behavior).

## Architecture

```
src/video_local/
├── __init__.py
├── gguf/                          ← generic GGUF infra (no S2V coupling)
│   ├── reader.py                  ← parse .gguf → {tensor_name: (raw_bytes, quant_type, shape)}
│   ├── dequant.py                 ← per-quant-type dequant kernels (torch ops, fp16 output)
│   │                                v1 implements: Q4_K_S, Q8_0  (fp16 skips dequant)
│   └── ops.py                     ← GGUFLinear nn.Module — dequantizes weight on forward, then matmul
├── sage_attn.py                   ← optional monkeypatch of torch.nn.functional.scaled_dot_product_attention
├── wan_s2v_runtime.py             ← LocalWanS2V: wraps Wan official wan.WanS2V pipeline,
│                                    swaps DiT nn.Linear → GGUFLinear when GGUF tier is active
├── manager.py                     ← VideoModelManager singleton + state machine
└── api.py                         ← /video/local/{status,voices,load,cancel,test_synthesize}

src/apps/comic_gen/video.py        ← VideoGenerator._build_model() dispatches on VIDEO_PROVIDER
src/apps/comic_gen/api.py          ← include local_video_router; EnvConfig adds
                                      VIDEO_PROVIDER, LOCAL_VIDEO_QUANT, USE_SAGE_ATTENTION

frontend/src/lib/api.ts            ← LocalVideoStatus, LocalVideoQuant types + API client
frontend/src/components/layout/GlobalStatusFooter.tsx   ← VIDEO row (mirrors LLM/T2I/I2I/TTS)
frontend/src/components/common/ModelSettingsModal.tsx   ← Video section: 3 cards (fp16/Q8_0/Q4_K_S)
```

**Boundaries**:
- `gguf/*` is model-agnostic. Tested in isolation with a small fixture GGUF (e.g. TinyLlama Q4_K_S).
- `wan_s2v_runtime.py` is the only S2V-specific code. Future I2V/Animate would add sibling files (`wan_i2v_runtime.py`, etc.) reusing `gguf/*` and `sage_attn.py`.
- `manager.py` follows the singleton + state machine pattern from `audio_local/manager.py` exactly.

## Components

### `gguf/reader.py`

- Single function: `parse_gguf(path) -> Dict[str, GGUFTensor]` where `GGUFTensor = (raw_bytes_view, quant_type, shape, ggml_type)`.
- Uses the `gguf` Python package (already on PyPI, no compile) for header parsing.
- Tensor data accessed via `mmap` so 13GB load takes seconds and the Python process doesn't double the file in RAM.
- Validates quant_type against the v1 whitelist (`F16`, `Q8_0`, `Q4_K_S`); raises `UnsupportedQuantError(quant_type)` otherwise.

### `gguf/dequant.py`

- One function per supported quant type, signature `dequant_q4ks(raw, shape, out_dtype=torch.float16) -> torch.Tensor` (returns a CUDA fp16 tensor).
- Logic ported from city96/ComfyUI-GGUF/`dequant.py`, written with pure torch ops (no llama.cpp at runtime).
- Tested for numerical equivalence against the `gguf` package's reference dequant.

### `gguf/ops.py:GGUFLinear`

- `nn.Linear`-shaped subclass holding `(quantized_bytes, quant_type, shape)` instead of an `nn.Parameter`.
- `forward(x)`: dequantize weight to GPU fp16 → run `F.linear(x, w, bias)` → discard the dequantized tensor (no caching, peak VRAM stays low).
- Bias handled normally (small, kept fp16 on GPU).
- Optional fast path: keep weights pinned int8/int4 buffers on GPU after first dequant (caches a small subset of hottest layers); v1 ships **without** this — every-forward dequant is the simplest correct baseline.

### `sage_attn.py`

- On module import: `try: import sageattention; ...` — if absent, log INFO and exit.
- Provides `enable_sage_attention()` / `disable_sage_attention()` to monkeypatch `torch.nn.functional.scaled_dot_product_attention` on demand.
- `LocalWanS2V` calls `enable_sage_attention()` if `USE_SAGE_ATTENTION=1` env var set.

### `wan_s2v_runtime.py:LocalWanS2V`

- Lazy-loads via `_inject_wan_paths()` (mirrors `cosyvoice_runtime.py`'s sys.path injection of the cloned `output/external/Wan2.2/` repo).
- Constructor takes `quant: Literal["fp16", "Q8_0", "Q4_K_S"]` from env.
- Construct `wan.WanS2V(...)` with stock args.
- If `quant != "fp16"`: download `QuantStack/Wan2.2-S2V-14B-GGUF` matching pattern, parse with `gguf/reader.py`, walk `pipe.noise_model` and replace every `nn.Linear` whose name has a corresponding GGUF tensor with `GGUFLinear`. Linear layers without a GGUF counterpart (norms, embeddings, audio encoder bits) stay fp16.
- If `USE_SAGE_ATTENTION=1`: enable Sage shim before first generate.
- `generate(image, audio, prompt, **kwargs)` calls through to `pipe.generate(...)` with the same kwargs as `output/s2v_validation_int8da.py` already established.

### `manager.py:VideoModelManager`

- Direct copy of `audio_local/manager.py` shape with class names swapped.
- States: `UNLOADED → LOADING/DOWNLOADING → READY → GENERATING → READY` (or `ERROR`).
- Tracks active `quant` so footer shows "VIDEO Wan2.2-S2V Q4_K_S".
- `unload()` drops the inner WanS2V pipe + clears CUDA cache; required when switching tiers.

### `api.py`

- `/video/local/status` — GET: state, quant, hf_id, progress, error
- `/video/local/load` — POST: trigger snapshot_download + wrap; idempotent if already READY
- `/video/local/cancel` — POST: drop model; clears ERROR
- `/video/local/test_synthesize` — POST: small synth using `output/external/Wan2.2/examples/i2v_input.JPG` + `examples/talk.wav` + a default prompt, saves to `output/test_video.mp4`

## Data flow

### Cold load (first switch to local)

1. UI Settings modal → click "Wan2.2-S2V (Q4_K_S Local)"
2. Frontend: `POST /config/env` with `{VIDEO_PROVIDER: "local", LOCAL_VIDEO_QUANT: "Q4_K_S"}` then `POST /video/local/load`
3. Manager: state → LOADING
4. snapshot_download `Wan-AI/Wan2.2-S2V-14B` (already cached, 46 GB) — instant
5. snapshot_download `QuantStack/Wan2.2-S2V-14B-GGUF` with `allow_patterns=["*Q4_K_S*"]` — ~13 GB on first run
6. Construct `wan.WanS2V` with stock args, but with `WanModel_S2V.from_pretrained` monkeypatched (or its 4 safetensors `load_state_dict` skipped) so the bf16 DiT shards never touch RAM. The architecture is built on `meta` device (zero memory) and the resulting `pipe.noise_model` has nn.Linear modules with placeholder weights.
7. Parse the GGUF file via `gguf/reader.py` (mmap, ~5 sec)
8. Walk `pipe.noise_model.named_modules()`, replace each `nn.Linear` whose name maps to a GGUF tensor with `GGUFLinear` (carrying mmap'd quantized bytes). Linear layers that have no GGUF counterpart load fp16 weights from the original safetensors shards on a per-tensor basis (small set: norms, embeddings, audio injector blocks).
9. Move quantized DiT to CUDA (~13 GB GPU)
10. If `USE_SAGE_ATTENTION=1`: `sage_attn.enable_sage_attention()`
11. Manager: state → READY

### Generate (single clip)

1. `POST /video/local/test_synthesize` (or `VideoGenerator.generate(frame)` from comic_gen)
2. Manager: state → GENERATING
3. Pass through to `pipe.generate(image, audio, prompt, max_area=480*480, infer_frames=21, sampling_steps=15, guide_scale=1.0, ...)` (params tuned for fast first-pass; user-overridable)
4. Each diffusion step: forward through `noise_model` → every `GGUFLinear.forward` dequantizes its weight on GPU → matmul → fp16 result; SDPA goes through Sage if enabled
5. VAE decode → mp4 written to `output/video_local/<id>.mp4`
6. Manager: state → READY

**VRAM budget at runtime (Q4_K_S + 480×480 + Sage)**:
- Quantized DiT resident: 13 GB
- T5 on CPU: 0 GB
- VAE on GPU during decode: 0.5 GB
- Per-step dequant scratch + activations: ~3-5 GB
- **Peak target: 16-18 GB** (vs un-quant 23.8 GB) — comfortable margin on 24 GB

## Testing

### Unit tests (`tests/video_local/`)

1. **GGUF reader**: parse a small public GGUF fixture (TinyLlama-Q4_K_S, ~700 MB), verify tensor count + shapes + quant types match the `gguf` package's view.
2. **dequant correctness**: round-trip — known fp16 weight → quant via `gguf` lib → our dequant → cosine similarity ≥ 0.99 (Q4_K_S) or ≥ 0.999 (Q8_0). This is the load-bearing correctness gate; wrong bit ops here = silent gibberish output.
3. **GGUFLinear ≈ nn.Linear**: same input through both; `atol=1e-2` for Q4_K_S, `1e-3` for Q8_0.
4. **Manager state machine**: UNLOADED → LOADING → READY → GENERATING → READY transitions; switching quant triggers unload+reload; cancel/error clean.
5. **Sage shim**: with sageattention not installed → no monkeypatch, original SDPA still bound; with installed → `F.scaled_dot_product_attention` returns the sage implementation's dispatch path.

### Integration tests (`tests/integration/`, `@pytest.mark.local_gpu`, skip in CI)

6. **End-to-end small synth**: full Q4_K_S load + 21-frame 480×480 generate using bundled `examples/i2v_input.JPG` + `examples/talk.wav`. Asserts: mp4 exists, duration ≈ expected, no NaN frames in decoded output.

### Manual acceptance

7. Run the integration test, open the resulting mp4, confirm visually that the talking head's mouth moves in sync with the audio. Quality must be visibly close to what un-quantized fp16 produces (subjective, eyeballed).

## Error handling

| Error | Detection | Behavior |
|---|---|---|
| GGUF download HTTPError | `snapshot_download` raises | Manager state → ERROR; surface message in footer |
| Unsupported quant_type encountered while parsing | `reader.py` whitelist check | Raise `UnsupportedQuantError`; UI shows "v1 supports {fp16, Q8_0, Q4_K_S}" |
| GGUF tensor name → model state_dict key mismatch | Pre-load dry run before any GPU work | Raise with diff list (missing/extra tensor names); user reports upstream |
| sageattention not installed when `USE_SAGE_ATTENTION=1` | ImportError on shim load | Log INFO, fall back to vanilla SDPA, do not block load |
| Forward produces NaN | Optional NaN guard on first generate (then disabled for perf) | Hard fail with "dequant kernel produced NaN — likely Q4_K_S bug" |
| OOM during generate | `torch.cuda.OutOfMemoryError` | Unload + state → ERROR; suggest "try Q4_K_S if Q8_0 OOM'd" |
| Stale GPU state when switching tiers | Manager.load detects different quant than current `_inner._quant` | Force unload before loading new tier (no shared weights) |

## Dependencies (incremental over current `requirements.txt`)

```
gguf>=0.6           # GGUF format parser, llama.cpp's official py package
# sageattention  ← OPTIONAL, install separately from prebuilt wheel
                   matching torch 2.5.x + cu121 + py3.12; not in requirements.txt
                   (mirrors flash_attn pattern — runtime-only, not a hard dep)
```

The 46 GB original snapshot is already in `output/models/LLM/hub/`. The 13 GB Q4_K_S GGUF will land alongside in the same cache (`models--QuantStack--Wan2.2-S2V-14B-GGUF/snapshots/<rev>/`). Disk impact: +13 GB per quant tier the user opts into.

## Out of scope / follow-ups

- **AOTI / `torch.export` precompile** — would shave the per-startup setup, write a separate spec.
- **Other Wan2.2 variants** — same infra, new model wrapper per variant.
- **Sage Attention auto-install helper** — if user demand justifies; for now manual install.
- **TeaCache / step skipping** — community speedup for diffusion sampling. Independent from GGUF; can layer on later.
- **Progress reporting via WebSocket** — current pattern is HTTP polling like image_local. Long generates (5+ min) might want streaming progress; defer.

## Open questions

None blocking implementation.
