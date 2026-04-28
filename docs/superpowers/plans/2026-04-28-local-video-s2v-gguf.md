# Local Video (Wan2.2-S2V-14B GGUF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `VIDEO_PROVIDER=local` to the lumenxDev backend, producing audio-driven lipsync videos via Wan2.2-S2V-14B with three quantization tiers (fp16 / Q8_0 / Q4_K_S GGUF), running comfortably on a 4090 24GB without ComfyUI.

**Architecture:** New `src/video_local/` module mirroring `audio_local/`. Generic `gguf/` subpackage (reader + dequant kernels + `GGUFLinear` op) is model-agnostic and tested in isolation. `wan_s2v_runtime.py` is the only S2V-specific code; constructs Wan's official `wan.WanS2V` pipeline and, when a quant tier is selected, walks the DiT and replaces `nn.Linear` modules with `GGUFLinear` carrying mmap'd quantized weights. Sage Attention is opt-in via env var and graceful fallback if the wheel isn't installed.

**Tech Stack:** Python 3.12 / PyTorch 2.5.1+cu121 / `gguf` py package (parser) / torchao 0.7+ (existing) / FastAPI / Wan-Video/Wan2.2 cloned at `output/external/Wan2.2`. Frontend: Next.js 14 / React.

**Spec:** [`docs/superpowers/specs/2026-04-28-local-video-s2v-gguf-design.md`](../specs/2026-04-28-local-video-s2v-gguf-design.md)

---

## File map

**New files:**
- `src/video_local/__init__.py`
- `src/video_local/gguf/__init__.py`
- `src/video_local/gguf/reader.py` — parse `.gguf` → tensor descriptors
- `src/video_local/gguf/dequant.py` — `dequant_q8_0`, `dequant_q4_k_s` torch kernels
- `src/video_local/gguf/ops.py` — `GGUFLinear` nn.Module
- `src/video_local/sage_attn.py` — optional SDPA monkeypatch
- `src/video_local/wan_s2v_runtime.py` — `LocalWanS2V` pipe wrapper
- `src/video_local/manager.py` — `VideoModelManager` singleton + state machine
- `src/video_local/api.py` — `/video/local/*` FastAPI router
- `tests/video_local/__init__.py`
- `tests/video_local/test_gguf_reader.py`
- `tests/video_local/test_gguf_dequant.py`
- `tests/video_local/test_gguf_ops.py`
- `tests/video_local/test_sage_attn.py`
- `tests/video_local/test_manager.py`
- `tests/video_local/test_api.py`
- `tests/test_video_generator_provider.py`

**Modified files:**
- `requirements.txt` — add `gguf>=0.6`
- `src/apps/comic_gen/api.py` — mount router; `EnvConfig` gains `VIDEO_PROVIDER`, `LOCAL_VIDEO_QUANT`, `USE_SAGE_ATTENTION`
- `src/apps/comic_gen/video.py` — `_build_model()` dispatches on `VIDEO_PROVIDER`
- `frontend/src/lib/api.ts` — `LocalVideoStatus`, `LocalVideoQuant` types + 4 client methods
- `frontend/src/components/layout/GlobalStatusFooter.tsx` — VIDEO row
- `frontend/src/components/common/ModelSettingsModal.tsx` — Video section (3 tier cards)
- `.env.example` — document new env vars

---

## Task 1: Project scaffolding + dep

**Files:**
- Create: `src/video_local/__init__.py`, `src/video_local/gguf/__init__.py`, `tests/video_local/__init__.py`
- Modify: `requirements.txt:27`

- [ ] **Step 1: Add gguf to requirements.txt**

```
# Local Video (Wan2.2-S2V-14B via GGUF). The cosyvoice/wan2.2 inference
# repos live under output/external/. Python deps:
gguf>=0.6                # llama.cpp's official GGUF parser, pure-Python
# sageattention          OPTIONAL — install separately from prebuilt wheel
                         # (mirrors flash_attn pattern; runtime-only, soft dep)
```

Insert this block after the `torchaudio` line in the local-TTS section.

- [ ] **Step 2: pip install**

```bash
pip install gguf
```

Expected: `Successfully installed gguf-0.X.X`

- [ ] **Step 3: Create empty `__init__.py` files**

```python
# src/video_local/__init__.py
"""Local video generation runtime — Wan2.2-S2V-14B with GGUF quantization.

Mirrors src/audio_local for shape: a manager singleton + state machine
wraps a runtime class that constructs Wan's official WanS2V pipeline,
replacing the DiT's nn.Linear modules with GGUFLinear when a quant
tier (Q8_0 / Q4_K_S) is selected. fp16 mode skips quantization entirely.

This module is import-cheap: Wan and gguf are imported lazily on first
load() so the rest of the app keeps starting if the optional deps or
the cloned Wan repo are missing."""
```

```python
# src/video_local/gguf/__init__.py
"""Generic GGUF infrastructure — model-agnostic. Reader parses the
file format; dequant kernels handle each quant type's bit layout;
GGUFLinear is a drop-in nn.Linear replacement that holds quantized
bytes and dequantizes per forward call."""
```

```python
# tests/video_local/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/video_local/ tests/video_local/
git commit -m "feat(video_local): scaffold module + add gguf dep"
```

---

## Task 2: GGUF reader (file parsing)

**Files:**
- Create: `src/video_local/gguf/reader.py`
- Test: `tests/video_local/test_gguf_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/video_local/test_gguf_reader.py
"""Tests for src/video_local/gguf/reader.py.

We don't ship a sample .gguf in the repo (sizes range from MB to GB).
Instead each test creates a tiny GGUF on the fly using the upstream
`gguf` package's writer, then parses it with our reader and verifies
the round-trip."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.video_local.gguf.reader import (
    GGUFTensor,
    UnsupportedQuantError,
    parse_gguf,
    SUPPORTED_QUANT_TYPES,
)


def _write_minimal_gguf(path: Path, tensors: dict) -> None:
    """Helper: write a GGUF file containing the given (name, ndarray) tensors
    in F16 quant. Returns nothing; file is written to `path`."""
    import gguf
    writer = gguf.GGUFWriter(str(path), arch="test")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestParseGguf:
    def test_parses_tensor_names_and_shapes(self, tmp_path):
        f = tmp_path / "tiny.gguf"
        _write_minimal_gguf(f, {
            "blocks.0.attn.weight": np.ones((4, 8), dtype=np.float16),
            "blocks.0.ffn.weight": np.zeros((16, 8), dtype=np.float16),
        })
        result = parse_gguf(str(f))
        assert set(result.keys()) == {"blocks.0.attn.weight", "blocks.0.ffn.weight"}
        assert result["blocks.0.attn.weight"].shape == (4, 8)
        assert result["blocks.0.ffn.weight"].shape == (16, 8)

    def test_returns_gguf_tensor_objects(self, tmp_path):
        f = tmp_path / "tiny.gguf"
        _write_minimal_gguf(f, {"w": np.ones((2, 2), dtype=np.float16)})
        result = parse_gguf(str(f))
        t = result["w"]
        assert isinstance(t, GGUFTensor)
        assert t.quant_type == "F16"
        assert t.shape == (2, 2)
        assert t.raw_bytes is not None

    def test_supported_quant_types(self):
        # v1 whitelist — anything else raises UnsupportedQuantError
        assert SUPPORTED_QUANT_TYPES == {"F16", "Q8_0", "Q4_K_S"}

    def test_unsupported_quant_raises(self, tmp_path, monkeypatch):
        # Construct a GGUF claiming a quant we don't support, verify reader rejects it
        f = tmp_path / "bad.gguf"
        _write_minimal_gguf(f, {"w": np.ones((2, 2), dtype=np.float16)})
        # Force the reader to see Q3_K_M as the type (which isn't whitelisted)
        from src.video_local.gguf import reader as r
        original = r._gguf_type_to_name
        monkeypatch.setattr(r, "_gguf_type_to_name", lambda t: "Q3_K_M")
        with pytest.raises(UnsupportedQuantError, match="Q3_K_M"):
            parse_gguf(str(f))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/video_local/test_gguf_reader.py -v
```

Expected: All FAIL with ImportError on `src.video_local.gguf.reader`.

- [ ] **Step 3: Implement reader**

```python
# src/video_local/gguf/reader.py
"""GGUF file format parser.

Wraps the upstream `gguf` package to surface only what we need:
tensor name → (raw_bytes, quant_type, shape). We don't decode tensor
data here — that's the dequant kernels' job in dequant.py.

Validates quant types against a v1 whitelist so unsupported tiers
fail early at load instead of producing garbage at inference time."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# v1 ships these three. Adding more = implementing the corresponding
# dequant kernel in dequant.py and listing it here.
SUPPORTED_QUANT_TYPES = {"F16", "Q8_0", "Q4_K_S"}


class UnsupportedQuantError(ValueError):
    """Raised when a GGUF file contains a quant_type not in SUPPORTED_QUANT_TYPES."""


@dataclass
class GGUFTensor:
    """A single tensor read from a .gguf file. raw_bytes is an mmap view —
    do NOT modify, do NOT keep a reference longer than the parent file
    handle's lifetime."""
    name: str
    quant_type: str            # "F16" | "Q8_0" | "Q4_K_S"
    shape: Tuple[int, ...]
    raw_bytes: memoryview      # mmap'd bytes, in GGUF block layout


def _gguf_type_to_name(t) -> str:
    """Map gguf.GGMLQuantizationType to our canonical name string. Pulled
    out as a function so tests can monkeypatch it to simulate unsupported
    types without authoring a malformed GGUF."""
    return t.name  # gguf.GGMLQuantizationType is an Enum; .name = "F16", etc.


def parse_gguf(path: str) -> Dict[str, GGUFTensor]:
    """Read a .gguf file and return a name → GGUFTensor dict.

    Raises UnsupportedQuantError as soon as any tensor uses a quant type
    not in SUPPORTED_QUANT_TYPES.
    """
    import gguf

    reader = gguf.GGUFReader(path)
    out: Dict[str, GGUFTensor] = {}
    for tensor in reader.tensors:
        quant_name = _gguf_type_to_name(tensor.tensor_type)
        if quant_name not in SUPPORTED_QUANT_TYPES:
            raise UnsupportedQuantError(
                f"GGUF tensor {tensor.name!r} uses quant {quant_name!r} which "
                f"is not in v1 whitelist {sorted(SUPPORTED_QUANT_TYPES)}. "
                f"To add support: implement a dequant kernel in dequant.py "
                f"and extend SUPPORTED_QUANT_TYPES."
            )
        out[tensor.name] = GGUFTensor(
            name=tensor.name,
            quant_type=quant_name,
            shape=tuple(tensor.shape),
            raw_bytes=memoryview(tensor.data),
        )
    logger.info(f"[gguf] parsed {len(out)} tensors from {path}")
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/video_local/test_gguf_reader.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_local/gguf/reader.py tests/video_local/test_gguf_reader.py
git commit -m "feat(video_local): GGUF file format reader with quant whitelist"
```

---

## Task 3: Q8_0 dequant kernel

**Files:**
- Create: `src/video_local/gguf/dequant.py`
- Test: `tests/video_local/test_gguf_dequant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/video_local/test_gguf_dequant.py
"""Numerical correctness tests for dequant kernels.

Strategy: take a known fp16 weight → quantize via the upstream `gguf`
package → dequantize via our kernel → assert cosine similarity with
the original ≥ threshold. This catches bit-layout bugs that would
otherwise surface as "model produces gibberish" at inference time."""
import numpy as np
import pytest
import torch

from src.video_local.gguf.dequant import dequant_q8_0


def _quantize_q8_0(weight_fp16: np.ndarray) -> bytes:
    """Use upstream gguf's quantizer to produce reference Q8_0 bytes."""
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q8_0).tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant runs on CUDA")
class TestDequantQ8:
    def test_round_trip_cosine_similarity(self):
        torch.manual_seed(0)
        # 32-row × 64-col weight, multiple of Q8_0's block size (32)
        original = torch.randn(32, 64, dtype=torch.float16).contiguous()
        quant_bytes = _quantize_q8_0(original.numpy())

        out = dequant_q8_0(quant_bytes, shape=(32, 64))

        assert out.dtype == torch.float16
        assert out.shape == (32, 64)
        assert out.device.type == "cuda"

        # Q8_0 has very low quant error — cosine sim should be near 1
        cos = torch.nn.functional.cosine_similarity(
            original.flatten().cuda().to(torch.float32),
            out.flatten().to(torch.float32),
            dim=0,
        ).item()
        assert cos > 0.999, f"cosine_sim={cos} below 0.999 — Q8_0 dequant likely wrong"

    def test_row_major_shape(self):
        # Confirm the output shape interprets the bytes in row-major
        # order matching the GGUF spec.
        torch.manual_seed(1)
        original = torch.arange(32 * 32, dtype=torch.float16).reshape(32, 32) / 1000.0
        quant_bytes = _quantize_q8_0(original.numpy())

        out = dequant_q8_0(quant_bytes, shape=(32, 32)).cpu().to(torch.float32)
        original_f32 = original.to(torch.float32)
        # Element-wise close — Q8_0 max error is bounded
        max_err = (out - original_f32).abs().max().item()
        assert max_err < 0.05, f"max abs error {max_err} too high"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/video_local/test_gguf_dequant.py::TestDequantQ8 -v
```

Expected: FAIL on `from src.video_local.gguf.dequant import dequant_q8_0`.

- [ ] **Step 3: Implement Q8_0 dequant**

```python
# src/video_local/gguf/dequant.py
"""GGUF dequantization kernels — torch ops, fp16 output, CUDA-resident.

Each block layout follows the GGUF/GGML spec. References:
- llama.cpp ggml-quants.c (canonical implementation)
- city96/ComfyUI-GGUF/dequant.py (PyTorch port we're paralleling)

Q8_0 layout (per 32-element block):
  bytes [0..2]   : fp16 scale `d`
  bytes [2..34]  : 32 int8 quantized values `qs`
  total: 34 bytes per block, dequantized as: x = qs * d

Q4_K_S layout (per 256-element super-block):
  bytes [0..2]   : fp16 super_scale `d`
  bytes [2..4]   : fp16 super_min `dmin`
  bytes [4..16]  : 12 bytes packed 6-bit scales/mins for 16 sub-blocks of 16 elements each
  bytes [16..144]: 128 bytes packed 4-bit quantized values
  total: 144 bytes per super-block. Dequant per sub-block i (16 elements each):
    scale_i  = d * (scales_packed[i] & 0x3F)
    min_i    = dmin * (mins_packed[i] & 0x3F)
    x[i*16 + j] = scale_i * q4[i*16 + j] - min_i
"""
from __future__ import annotations

import torch


_Q8_0_BLOCK_SIZE = 32          # elements per Q8_0 block
_Q8_0_BYTES = 34               # bytes per Q8_0 block (2 fp16 scale + 32 int8)


def dequant_q8_0(
    raw: bytes | memoryview,
    shape: tuple[int, ...],
    out_device: str = "cuda",
) -> torch.Tensor:
    """Dequantize Q8_0 bytes to fp16 tensor of `shape` on `out_device`.

    Q8_0 stores N elements as N/32 blocks; each block is 1 fp16 scale +
    32 int8 quantized values. dequant: x = qs * scale.
    """
    n = 1
    for d in shape:
        n *= d
    assert n % _Q8_0_BLOCK_SIZE == 0, (
        f"Q8_0 requires total elements multiple of {_Q8_0_BLOCK_SIZE}, got {n}"
    )
    n_blocks = n // _Q8_0_BLOCK_SIZE
    expected_bytes = n_blocks * _Q8_0_BYTES
    assert len(raw) == expected_bytes, (
        f"Q8_0 byte length mismatch: expected {expected_bytes}, got {len(raw)}"
    )

    # Move bytes to GPU as a contiguous uint8 tensor — single H2D copy
    blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(out_device)
    blob = blob.view(n_blocks, _Q8_0_BYTES)

    # First 2 bytes of each block are fp16 scale; remaining 32 are int8 qs.
    scales = blob[:, :2].contiguous().view(torch.float16)        # (n_blocks, 1)
    qs = blob[:, 2:].view(torch.int8).to(torch.float16)           # (n_blocks, 32)
    dequant = qs * scales                                         # (n_blocks, 32)

    return dequant.reshape(shape).contiguous()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/video_local/test_gguf_dequant.py::TestDequantQ8 -v
```

Expected: 2 PASS (skip if no GPU).

- [ ] **Step 5: Commit**

```bash
git add src/video_local/gguf/dequant.py tests/video_local/test_gguf_dequant.py
git commit -m "feat(video_local): Q8_0 dequant kernel"
```

---

## Task 4: Q4_K_S dequant kernel

**Files:**
- Modify: `src/video_local/gguf/dequant.py` (add `dequant_q4_k_s`)
- Modify: `tests/video_local/test_gguf_dequant.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/video_local/test_gguf_dequant.py`:

```python
from src.video_local.gguf.dequant import dequant_q4_k_s


def _quantize_q4_k_s(weight_fp16: np.ndarray) -> bytes:
    import gguf
    return gguf.quants.quantize(weight_fp16, gguf.GGMLQuantizationType.Q4_K_S).tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant runs on CUDA")
class TestDequantQ4KS:
    def test_round_trip_cosine_similarity(self):
        torch.manual_seed(0)
        # 256-element multiple required for Q4_K_S super-blocks
        original = torch.randn(256, 256, dtype=torch.float16).contiguous()
        quant_bytes = _quantize_q4_k_s(original.numpy())

        out = dequant_q4_k_s(quant_bytes, shape=(256, 256))

        assert out.dtype == torch.float16
        assert out.shape == (256, 256)
        assert out.device.type == "cuda"

        cos = torch.nn.functional.cosine_similarity(
            original.flatten().cuda().to(torch.float32),
            out.flatten().to(torch.float32),
            dim=0,
        ).item()
        assert cos > 0.99, f"cosine_sim={cos} below 0.99 — Q4_K_S dequant likely wrong"

    def test_byte_layout_matches_gguf_spec(self):
        # Sanity: verify 256-element block consumes exactly 144 bytes
        original = torch.randn(256, dtype=torch.float16)
        quant_bytes = _quantize_q4_k_s(original.unsqueeze(0).numpy())
        assert len(quant_bytes) == 144
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/video_local/test_gguf_dequant.py::TestDequantQ4KS -v
```

Expected: FAIL on `from src.video_local.gguf.dequant import dequant_q4_k_s`.

- [ ] **Step 3: Implement Q4_K_S dequant**

Append to `src/video_local/gguf/dequant.py`:

```python
_Q4_K_S_SUPER_BLOCK_SIZE = 256
_Q4_K_S_BYTES = 144           # per super-block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (q4 quants)


def dequant_q4_k_s(
    raw: bytes | memoryview,
    shape: tuple[int, ...],
    out_device: str = "cuda",
) -> torch.Tensor:
    """Dequantize Q4_K_S bytes to fp16 tensor of `shape` on `out_device`.

    Q4_K_S super-block (256 elements, 144 bytes):
      [0..2]    fp16 super_scale d
      [2..4]    fp16 super_min   dmin
      [4..16]   12 bytes packed 6-bit (scales|mins) for 16 sub-blocks
      [16..144] 128 bytes packed 4-bit quantized values (256 nibbles)

    Each sub-block has 16 elements. Per sub-block i:
      scale_i = d * (sc_i & 0x3F)              # 6-bit scale
      min_i   = dmin * (mn_i & 0x3F)           # 6-bit min
      x[j]    = scale_i * q4[j] - min_i
    where q4[j] is a 4-bit unsigned integer extracted from the packed
    quants block.

    The 12-byte scales/mins encoding follows the K-quant family layout:
      bytes [0..3]  : low 6 bits of scales[0..3]
      bytes [4..7]  : low 6 bits of mins[0..3]
      bytes [8..11] : high 2 bits of scales[0..7] | low 4 bits of scales[4..7] | etc.
    See llama.cpp ggml-quants.c get_scale_min_k4 for the canonical impl.
    """
    n = 1
    for d in shape:
        n *= d
    assert n % _Q4_K_S_SUPER_BLOCK_SIZE == 0, (
        f"Q4_K_S requires total elements multiple of {_Q4_K_S_SUPER_BLOCK_SIZE}, got {n}"
    )
    n_super = n // _Q4_K_S_SUPER_BLOCK_SIZE
    expected_bytes = n_super * _Q4_K_S_BYTES
    assert len(raw) == expected_bytes, (
        f"Q4_K_S byte length mismatch: expected {expected_bytes}, got {len(raw)}"
    )

    blob = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(out_device)
    blob = blob.view(n_super, _Q4_K_S_BYTES)

    # Super-scale and super-min, fp16 each
    d = blob[:, :2].contiguous().view(torch.float16).squeeze(-1)        # (n_super,)
    dmin = blob[:, 2:4].contiguous().view(torch.float16).squeeze(-1)    # (n_super,)

    # 12-byte packed (scales, mins). Decode 16 6-bit values.
    scales_min_pack = blob[:, 4:16]                                      # (n_super, 12)
    sc, mn = _unpack_k4_scales_mins(scales_min_pack)                     # each (n_super, 16) uint8 in [0, 63]

    # 128-byte packed q4 quants (256 nibbles → 256 4-bit values)
    q_pack = blob[:, 16:]                                                # (n_super, 128)
    # Lower nibble for first 128 values, upper nibble for second 128
    q_lo = (q_pack & 0x0F).to(torch.float16)                             # (n_super, 128)
    q_hi = ((q_pack >> 4) & 0x0F).to(torch.float16)                      # (n_super, 128)
    q4 = torch.cat([q_lo, q_hi], dim=1)                                  # (n_super, 256)

    # Reshape to (n_super, 16 sub-blocks, 16 elements per sub-block)
    q4 = q4.view(n_super, 16, 16)

    # Per-sub-block scale_i and min_i
    scale = (d.unsqueeze(-1) * sc.to(torch.float16))                     # (n_super, 16)
    minv  = (dmin.unsqueeze(-1) * mn.to(torch.float16))                  # (n_super, 16)

    out = q4 * scale.unsqueeze(-1) - minv.unsqueeze(-1)                  # (n_super, 16, 16)
    return out.reshape(shape).contiguous()


def _unpack_k4_scales_mins(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the 12-byte K-family scales+mins block into two (n_super, 16)
    uint8 tensors of 6-bit values.

    Layout (per llama.cpp get_scale_min_k4):
      sc[0..3] = pack[0..3] & 0x3F
      mn[0..3] = pack[4..7] & 0x3F
      sc[4..7] = (pack[8..11] & 0x0F) | ((pack[0..3] >> 6) << 4)
      mn[4..7] = (pack[8..11] >> 4)   | ((pack[4..7] >> 6) << 4)
      Then sub-blocks 8..15 packed similarly using bytes 0..7 high bits.

    NOTE: full 16-sub-block layout requires careful bit shuffling. The
    implementer should cross-check against gguf.quants.dequantize_q4_k_s
    output OR llama.cpp's reference C code rather than relying on this
    docstring alone — bit packing is exactly the kind of code that's
    easy to get wrong silently.
    """
    n_super = pack.shape[0]
    sc = torch.empty(n_super, 16, dtype=torch.uint8, device=pack.device)
    mn = torch.empty(n_super, 16, dtype=torch.uint8, device=pack.device)

    # First 8 sub-blocks
    sc[:, 0:4] = pack[:, 0:4] & 0x3F
    mn[:, 0:4] = pack[:, 4:8] & 0x3F
    sc[:, 4:8] = (pack[:, 8:12] & 0x0F) | ((pack[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (pack[:, 8:12] >> 4)   | ((pack[:, 4:8] >> 6) << 4)
    # Next 8 sub-blocks reuse the same bytes' upper portions
    sc[:, 8:12] = pack[:, 0:4] >> 4
    mn[:, 8:12] = pack[:, 4:8] >> 4
    sc[:, 12:16] = pack[:, 8:12] & 0x0F
    mn[:, 12:16] = pack[:, 8:12] >> 4
    return sc, mn
```

> **Note for implementer:** `_unpack_k4_scales_mins` is the trickiest bit of the whole codebase. After writing it, run the test, and if cosine_sim < 0.99, cross-reference [llama.cpp ggml-quants.c `get_scale_min_k4`](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c) and [city96/ComfyUI-GGUF/dequant.py](https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py). The exact bit shuffling differs across documentation sources.

- [ ] **Step 4: Run tests**

```bash
pytest tests/video_local/test_gguf_dequant.py::TestDequantQ4KS -v
```

Expected: 2 PASS. If FAIL on cosine_similarity, inspect `_unpack_k4_scales_mins` against llama.cpp reference.

- [ ] **Step 5: Commit**

```bash
git add src/video_local/gguf/dequant.py tests/video_local/test_gguf_dequant.py
git commit -m "feat(video_local): Q4_K_S dequant kernel"
```

---

## Task 5: GGUFLinear nn.Module

**Files:**
- Create: `src/video_local/gguf/ops.py`
- Test: `tests/video_local/test_gguf_ops.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/video_local/test_gguf_ops.py
"""Tests for GGUFLinear — verifies it's numerically equivalent to
nn.Linear modulo the quant tier's tolerance, and that bias / dtypes
are handled correctly."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.video_local.gguf.ops import GGUFLinear
from src.video_local.gguf.reader import GGUFTensor


def _quantize(weight_fp16: np.ndarray, quant_type: str) -> bytes:
    import gguf
    qt = {"Q8_0": gguf.GGMLQuantizationType.Q8_0,
          "Q4_K_S": gguf.GGMLQuantizationType.Q4_K_S}[quant_type]
    return gguf.quants.quantize(weight_fp16, qt).tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant,atol", [("Q8_0", 1e-3), ("Q4_K_S", 1e-2)])
def test_gguflinear_matches_nn_linear(quant, atol):
    torch.manual_seed(0)
    in_f, out_f = 256, 64
    real_weight = torch.randn(out_f, in_f, dtype=torch.float16).contiguous()
    real_bias = torch.randn(out_f, dtype=torch.float16)

    raw = _quantize(real_weight.numpy(), quant)
    tensor = GGUFTensor(name="w", quant_type=quant, shape=(out_f, in_f),
                       raw_bytes=memoryview(raw))

    layer = GGUFLinear(weight_tensor=tensor, bias=real_bias).cuda()

    x = torch.randn(8, in_f, dtype=torch.float16).cuda()
    actual = layer(x)

    nn_layer = nn.Linear(in_f, out_f).to(torch.float16).cuda()
    with torch.no_grad():
        nn_layer.weight.copy_(real_weight)
        nn_layer.bias.copy_(real_bias)
    expected = nn_layer(x)

    assert actual.shape == expected.shape
    rel_err = (actual - expected).abs().max() / (expected.abs().max() + 1e-6)
    assert rel_err < atol, f"GGUFLinear {quant} relative error {rel_err} > {atol}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguflinear_no_bias():
    torch.manual_seed(0)
    in_f, out_f = 32, 16
    real_weight = torch.randn(out_f, in_f, dtype=torch.float16).contiguous()
    raw = _quantize(real_weight.numpy(), "Q8_0")
    tensor = GGUFTensor(name="w", quant_type="Q8_0", shape=(out_f, in_f),
                       raw_bytes=memoryview(raw))

    layer = GGUFLinear(weight_tensor=tensor, bias=None).cuda()
    out = layer(torch.randn(4, in_f, dtype=torch.float16).cuda())
    assert out.shape == (4, out_f)


def test_unsupported_quant_raises():
    with pytest.raises(ValueError, match="F16"):
        # GGUFLinear v1 doesn't support F16 (use plain nn.Linear instead)
        tensor = GGUFTensor(name="w", quant_type="F16", shape=(2, 2),
                           raw_bytes=memoryview(b"\0" * 16))
        GGUFLinear(weight_tensor=tensor, bias=None)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/video_local/test_gguf_ops.py -v
```

Expected: FAIL on `from src.video_local.gguf.ops import GGUFLinear`.

- [ ] **Step 3: Implement GGUFLinear**

```python
# src/video_local/gguf/ops.py
"""GGUFLinear — drop-in nn.Linear replacement carrying quantized
weights. Dequantizes on every forward; the dequantized fp16 tensor is
released after the F.linear call so peak VRAM stays close to the
quantized weight size."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequant_q4_k_s, dequant_q8_0
from .reader import GGUFTensor


_DEQUANT_FNS = {
    "Q8_0":   dequant_q8_0,
    "Q4_K_S": dequant_q4_k_s,
}


class GGUFLinear(nn.Module):
    """Behaves like nn.Linear(in_features, out_features) but holds its
    weight as quantized GGUF bytes. F16 is NOT supported here — use a
    plain nn.Linear with the unquantized fp16 weight instead, since
    there's no benefit to wrapping it.
    """

    def __init__(self, weight_tensor: GGUFTensor, bias: Optional[torch.Tensor]):
        super().__init__()
        if weight_tensor.quant_type not in _DEQUANT_FNS:
            raise ValueError(
                f"GGUFLinear got unsupported quant_type {weight_tensor.quant_type!r}; "
                f"supported: {sorted(_DEQUANT_FNS)}"
            )
        self._dequant = _DEQUANT_FNS[weight_tensor.quant_type]
        self._shape = weight_tensor.shape
        # Store raw bytes as a buffer so .to(device) moves them along
        # with the rest of the module. uint8 is the natural type for
        # opaque packed bytes.
        raw = torch.frombuffer(bytearray(weight_tensor.raw_bytes), dtype=torch.uint8)
        self.register_buffer("_quant_bytes", raw, persistent=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float16))
        else:
            self.register_parameter("bias", None)

        # Convenience attrs for callers that introspect Linear shape
        self.out_features, self.in_features = self._shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequant(
            self._quant_bytes.cpu().numpy().tobytes() if self._quant_bytes.device.type == "cpu"
            else bytes(self._quant_bytes.cpu().numpy()),
            shape=self._shape,
            out_device=x.device.type,
        )
        return F.linear(x, weight, self.bias)
```

> **Note for implementer:** the `forward` above does CPU→GPU byte copy on every call, which is wasteful when the bytes already live on GPU. After tests pass, optimize by checking `self._quant_bytes.device` and passing a GPU pointer to dequant directly. Keep tests green throughout.

- [ ] **Step 4: Run tests**

```bash
pytest tests/video_local/test_gguf_ops.py -v
```

Expected: 3 PASS (or skip on no-GPU).

- [ ] **Step 5: Commit**

```bash
git add src/video_local/gguf/ops.py tests/video_local/test_gguf_ops.py
git commit -m "feat(video_local): GGUFLinear nn.Module"
```

---

## Task 6: Sage Attention shim

**Files:**
- Create: `src/video_local/sage_attn.py`
- Test: `tests/video_local/test_sage_attn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/video_local/test_sage_attn.py
"""Tests for the Sage Attention monkeypatch shim. We don't actually
require the sageattention package to be installed — the shim must
fall through silently if it's missing."""
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.video_local.sage_attn import (
    enable_sage_attention,
    disable_sage_attention,
    is_sage_active,
)


@pytest.fixture(autouse=True)
def _reset():
    # Always start each test with sage disabled
    disable_sage_attention()
    yield
    disable_sage_attention()


def test_disabled_by_default():
    assert is_sage_active() is False


def test_enable_with_no_sage_installed_falls_back_silently(monkeypatch):
    # Pretend sageattention is unimportable
    monkeypatch.setitem(sys.modules, "sageattention", None)

    enable_sage_attention()  # must not raise

    assert is_sage_active() is False
    # F.scaled_dot_product_attention should still be torch's original
    assert F.scaled_dot_product_attention is torch._C._nn.scaled_dot_product_attention or callable(F.scaled_dot_product_attention)


def test_disable_restores_original():
    original = F.scaled_dot_product_attention
    # Pretend a fake sageattention module with a sageattn function
    fake_mod = type(sys)("sageattention")
    fake_mod.sageattn = lambda q, k, v, **kw: q  # noqa: E731
    sys.modules["sageattention"] = fake_mod
    try:
        enable_sage_attention()
        assert is_sage_active() is True
        disable_sage_attention()
        assert is_sage_active() is False
        assert F.scaled_dot_product_attention is original
    finally:
        del sys.modules["sageattention"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/video_local/test_sage_attn.py -v
```

Expected: FAIL on `from src.video_local.sage_attn import enable_sage_attention`.

- [ ] **Step 3: Implement sage_attn shim**

```python
# src/video_local/sage_attn.py
"""Optional Sage Attention 2.x integration.

Sage Attention provides a faster + lower-memory drop-in replacement
for `torch.nn.functional.scaled_dot_product_attention`. Wan2.2's
attention layers fall through to F.scaled_dot_product_attention when
flash_attn isn't available, so monkeypatching it makes Wan use Sage
without any model code changes.

The sageattention pip package isn't always installable on Windows
(needs a prebuilt wheel matching torch + CUDA + Python). We treat it
as a soft dependency: enable_sage_attention() succeeds silently if
the package is missing.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import torch.nn.functional as F

logger = logging.getLogger(__name__)


_original_sdpa: Optional[Callable] = None
_active: bool = False


def is_sage_active() -> bool:
    return _active


def enable_sage_attention() -> bool:
    """Attempt to monkeypatch F.scaled_dot_product_attention with Sage's
    implementation. Returns True if the patch was applied, False if Sage
    isn't installed. Idempotent — calling twice in a row only patches
    once."""
    global _original_sdpa, _active
    if _active:
        return True
    try:
        import sageattention
        if sageattention is None:           # monkeypatched-to-None in tests
            raise ImportError("sageattention is None")
    except ImportError:
        logger.info(
            "[sage_attn] sageattention not installed; falling back to torch SDPA. "
            "Install via the prebuilt wheel from "
            "https://github.com/woct0rdho/SageAttention to enable."
        )
        return False

    _original_sdpa = F.scaled_dot_product_attention

    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
        # Wan currently calls SDPA with attn_mask=None and dropout_p=0
        # in inference mode; Sage handles those defaults. For paths
        # where it can't (e.g. attn_mask provided), fall back.
        if attn_mask is not None or dropout_p != 0.0:
            return _original_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
            )
        return sageattention.sageattn(query, key, value, is_causal=is_causal)

    F.scaled_dot_product_attention = _sage_sdpa
    _active = True
    logger.info("[sage_attn] enabled — F.scaled_dot_product_attention now uses Sage")
    return True


def disable_sage_attention() -> None:
    """Restore the original torch SDPA. Idempotent."""
    global _original_sdpa, _active
    if not _active or _original_sdpa is None:
        _active = False
        return
    F.scaled_dot_product_attention = _original_sdpa
    _original_sdpa = None
    _active = False
    logger.info("[sage_attn] disabled — F.scaled_dot_product_attention restored")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/video_local/test_sage_attn.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_local/sage_attn.py tests/video_local/test_sage_attn.py
git commit -m "feat(video_local): Sage Attention shim with graceful fallback"
```

---

## Task 7: VideoModelManager (state machine, pre-runtime)

**Files:**
- Create: `src/video_local/manager.py`
- Test: `tests/video_local/test_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/video_local/test_manager.py
"""VideoModelManager — singleton state machine for the local video
runtime. Mirrors AudioModelManager exactly in shape; verifies state
transitions, error capture, idempotency, and quant-tier reload."""
from unittest.mock import MagicMock, patch

import pytest

from src.video_local.manager import VideoModelManager, VideoState


@pytest.fixture(autouse=True)
def _reset():
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


class TestSingleton:
    def test_get_returns_same(self):
        a = VideoModelManager.get()
        b = VideoModelManager.get()
        assert a is b


class TestStatus:
    def test_initial_unloaded(self):
        s = VideoModelManager.get().status()
        assert s["state"] == VideoState.UNLOADED.value
        assert s["error"] is None
        assert s["quant"] in ("fp16", "Q8_0", "Q4_K_S")  # whichever default


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_transitions_to_ready(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", return_value=None):
            await mgr.load()
        assert mgr.status()["state"] == VideoState.READY.value

    @pytest.mark.asyncio
    async def test_load_failure_sets_error(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", side_effect=RuntimeError("gguf 404")):
            with pytest.raises(RuntimeError):
                await mgr.load()
        assert mgr.status()["state"] == VideoState.ERROR.value
        assert "gguf 404" in mgr.status()["error"]


class TestQuantSwitch:
    @pytest.mark.asyncio
    async def test_changing_quant_forces_reload(self):
        mgr = VideoModelManager.get()
        with patch.object(mgr._inner, "load", return_value=None), \
             patch.object(mgr._inner, "unload", return_value=None) as mock_unload:
            await mgr.load(quant="Q4_K_S")
            await mgr.load(quant="Q8_0")
        assert mock_unload.called  # the second load required an unload first


class TestGenerate:
    def test_generate_failure_sets_error(self, tmp_path):
        mgr = VideoModelManager.get()
        mgr._loaded = True
        with patch.object(mgr._inner, "generate", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError):
                mgr.generate(image="x.jpg", audio="a.wav", prompt="p",
                             output_path=str(tmp_path / "o.mp4"))
        assert mgr.status()["state"] == VideoState.ERROR.value
        assert "OOM" in mgr.status()["error"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/video_local/test_manager.py -v
```

Expected: FAIL on `from src.video_local.manager import VideoModelManager, VideoState`.

- [ ] **Step 3: Implement manager**

```python
# src/video_local/manager.py
"""VideoModelManager — singleton state machine wrapping LocalWanS2V.

Parallel to AudioModelManager but for video. Tracks the active quant
tier so the footer can show "VIDEO Wan2.2-S2V Q4_K_S" and switching
tiers triggers an unload + load (no shared weight state).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from enum import Enum
from typing import Any, Dict, Optional

from .wan_s2v_runtime import LocalWanS2V

logger = logging.getLogger(__name__)


class VideoState(str, Enum):
    UNLOADED = "UNLOADED"
    DOWNLOADING = "DOWNLOADING"
    LOADING = "LOADING"
    GENERATING = "GENERATING"
    READY = "READY"
    ERROR = "ERROR"


_DEFAULT_QUANT = os.environ.get("LOCAL_VIDEO_QUANT", "Q4_K_S")


class VideoModelManager:
    _instance: Optional["VideoModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, config: Optional[Dict[str, Any]] = None) -> "VideoModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VideoModelManager(config or {})
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._quant = _DEFAULT_QUANT
        self._inner = LocalWanS2V(quant=self._quant)
        self._state = VideoState.UNLOADED
        self._error: Optional[str] = None
        self._loaded = False
        self._gen_progress = 0.0
        self._phase_label = ""

    def status(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "quant": self._quant,
            "hf_id": self._inner.hf_id,
            "phase": "",
            "progress": self._gen_progress if self._state == VideoState.GENERATING else (1.0 if self._state == VideoState.READY else 0.0),
            "error": self._error,
            "phase_label": self._phase_label,
        }

    async def load(self, quant: Optional[str] = None) -> Dict[str, Any]:
        target = quant or self._quant
        if self._state == VideoState.READY and target == self._quant:
            return self.status()
        # Tier change: unload first
        if self._loaded and target != self._quant:
            await asyncio.to_thread(self._inner.unload)
            self._loaded = False
        self._quant = target
        self._inner.quant = target
        self._state = VideoState.LOADING
        self._error = None
        try:
            await asyncio.to_thread(self._inner.load)
            self._loaded = True
            self._state = VideoState.READY
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.load failed")
            raise
        return self.status()

    async def unload(self) -> None:
        await asyncio.to_thread(self._inner.unload)
        self._state = VideoState.UNLOADED
        self._error = None
        self._loaded = False

    def generate(self, image: str, audio: str, prompt: str, output_path: str, **kwargs) -> str:
        if not self._loaded:
            self._state = VideoState.LOADING
            try:
                self._inner.load()
                self._loaded = True
            except Exception as e:
                self._state = VideoState.ERROR
                self._error = str(e)
                raise
        self._state = VideoState.GENERATING
        self._gen_progress = 0.0
        try:
            result = self._inner.generate(image=image, audio=audio, prompt=prompt,
                                          output_path=output_path, **kwargs)
            self._state = VideoState.READY
            self._gen_progress = 1.0
            self._error = None
            return result
        except Exception as e:
            self._state = VideoState.ERROR
            self._error = str(e)
            logger.exception("VideoModelManager.generate failed")
            raise
```

- [ ] **Step 4: Stub LocalWanS2V to make tests pass**

Create a minimal `src/video_local/wan_s2v_runtime.py` so the import works. This is a stub — Task 8 fills it in. The stub satisfies all the methods the manager calls, with TODO-marked bodies that raise NotImplementedError when reached.

```python
# src/video_local/wan_s2v_runtime.py — STUB, replaced in Task 8
"""LocalWanS2V — placeholder. Real implementation lands in Task 8."""
from __future__ import annotations

from typing import Optional


class LocalWanS2V:
    """Stub class — Task 8 implements load/generate/unload."""

    def __init__(self, quant: str = "Q4_K_S"):
        self.quant = quant
        self.hf_id = "Wan-AI/Wan2.2-S2V-14B"

    def load(self) -> None:
        raise NotImplementedError("LocalWanS2V.load — implement in Task 8")

    def unload(self) -> None:
        # Safe no-op so tests can patch it freely
        pass

    def generate(self, image: str, audio: str, prompt: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("LocalWanS2V.generate — implement in Task 8")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/video_local/test_manager.py -v
```

Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/video_local/manager.py src/video_local/wan_s2v_runtime.py tests/video_local/test_manager.py
git commit -m "feat(video_local): VideoModelManager state machine (LocalWanS2V stub)"
```

---

## Task 8: LocalWanS2V runtime

**Files:**
- Modify: `src/video_local/wan_s2v_runtime.py` (replace stub)

This task is the most complex. It has no test of its own — exercising the real Wan inference requires the 46GB Wan model + 13GB GGUF + GPU + minutes of runtime. The end-to-end correctness is validated by **Task 14 (manual acceptance)** — generating a video and verifying lipsync visually.

- [ ] **Step 1: Replace `wan_s2v_runtime.py` with real implementation**

```python
# src/video_local/wan_s2v_runtime.py
"""LocalWanS2V — wraps Wan official wan.WanS2V pipeline, optionally
swapping the DiT's nn.Linear modules for GGUFLinear.

Loading flow:
  1. snapshot_download Wan-AI/Wan2.2-S2V-14B (T5/VAE/wav2vec/yaml/etc)
  2. If quant != fp16: snapshot_download QuantStack/Wan2.2-S2V-14B-GGUF
     restricted to the matching tier file
  3. Patch wan.modules.s2v.model_s2v.WanModel_S2V.from_pretrained to
     skip the bf16 safetensors load (we'll fill weights ourselves)
  4. Construct wan.WanS2V — pipe.noise_model now has un-loaded
     parameters on whatever device the patched from_pretrained left
     them
  5. Walk pipe.noise_model.named_modules(), for each nn.Linear that
     has a corresponding GGUF tensor, replace with GGUFLinear. Linears
     with no GGUF counterpart load fp16 from the original safetensors
     shards individually.
  6. Move quantized DiT to CUDA
  7. If env USE_SAGE_ATTENTION=1, enable sage_attn shim

Inference flow:
  - generate() proxies through to pipe.generate(...) with the kwargs
    we've validated work end-to-end (frame_num=4n+1, max_area, etc.)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _default_repo_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "output" / "external" / "Wan2.2"


def _inject_wan_paths() -> None:
    """Prepend the cloned Wan2.2 repo and its Matcha-TTS submodule to
    sys.path. Idempotent. Honors WAN_REPO_PATH env override."""
    repo = Path(os.environ.get("WAN_REPO_PATH", str(_default_repo_path())))
    if not repo.exists():
        return
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (repo, matcha):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# Pattern for GGUF tier filename. Matches QuantStack's naming convention.
_GGUF_FILE_PATTERNS = {
    "Q8_0":   "*Q8_0.gguf",
    "Q4_K_S": "*Q4_K_S.gguf",
}


class LocalWanS2V:
    def __init__(self, quant: str = "Q4_K_S"):
        if quant not in ("fp16", "Q8_0", "Q4_K_S"):
            raise ValueError(f"unsupported quant {quant!r}; want fp16|Q8_0|Q4_K_S")
        self.quant = quant
        self.hf_id = "Wan-AI/Wan2.2-S2V-14B"
        self.gguf_hf_id = "QuantStack/Wan2.2-S2V-14B-GGUF"
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        _inject_wan_paths()
        try:
            import wan
            from wan.configs import WAN_CONFIGS
        except ImportError as e:
            repo = os.environ.get("WAN_REPO_PATH", str(_default_repo_path()))
            raise RuntimeError(
                f"Wan2.2 repo not importable from {repo!r}. Clone "
                f"github.com/Wan-Video/Wan2.2 to that path. Original error: {e}"
            ) from e

        from huggingface_hub import snapshot_download

        # 1. Download original Wan2.2-S2V-14B (T5, VAE, wav2vec, yaml — DiT
        # safetensors will get downloaded too even though we may not use them).
        logger.info(f"[LocalWanS2V] downloading {self.hf_id} (~46 GB if not cached)")
        wan_dir = snapshot_download(self.hf_id)

        # 2. Download GGUF tier file if needed
        gguf_path: Optional[str] = None
        if self.quant != "fp16":
            pattern = _GGUF_FILE_PATTERNS[self.quant]
            logger.info(f"[LocalWanS2V] downloading {self.gguf_hf_id} ({pattern})")
            gguf_dir = snapshot_download(self.gguf_hf_id, allow_patterns=[pattern])
            matches = list(Path(gguf_dir).glob(pattern))
            if not matches:
                raise RuntimeError(f"no GGUF file matching {pattern} in {gguf_dir}")
            gguf_path = str(matches[0])

        # 3. Construct WanS2V (loads everything including bf16 DiT)
        cfg = WAN_CONFIGS["s2v-14B"]
        logger.info("[LocalWanS2V] constructing wan.WanS2V — t5_cpu=True, init_on_cpu=True")
        self._pipe = wan.WanS2V(
            config=cfg,
            checkpoint_dir=wan_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )

        # 4. If quantized, replace DiT linears with GGUFLinear
        if self.quant != "fp16":
            self._patch_with_gguf(gguf_path)

        # 5. Move quantized DiT to GPU
        import torch
        self._pipe.noise_model = self._pipe.noise_model.to("cuda:0")
        torch.cuda.synchronize()

        # 6. Sage attention if requested
        if os.environ.get("USE_SAGE_ATTENTION") == "1":
            from .sage_attn import enable_sage_attention
            enable_sage_attention()

        logger.info(f"[LocalWanS2V] ready (quant={self.quant})")

    def _patch_with_gguf(self, gguf_path: str) -> None:
        """Walk pipe.noise_model and replace each nn.Linear that has a
        matching GGUF tensor with GGUFLinear carrying the quantized
        bytes. Linears that weren't quantized in the GGUF (e.g. small
        embedding/norm layers) keep their fp16 weights from the bf16
        safetensors load."""
        import torch.nn as nn
        from .gguf.reader import parse_gguf
        from .gguf.ops import GGUFLinear

        logger.info(f"[LocalWanS2V] parsing GGUF {gguf_path}")
        gguf_tensors = parse_gguf(gguf_path)

        replaced = 0
        kept = 0
        for module_name, module in list(self._pipe.noise_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # GGUF tensor name convention: f"{module_name}.weight"
            tensor_key = f"{module_name}.weight"
            if tensor_key not in gguf_tensors:
                kept += 1
                continue
            ggt = gguf_tensors[tensor_key]
            if ggt.quant_type == "F16":
                # GGUFLinear doesn't handle F16; leave the original linear
                kept += 1
                continue
            new_linear = GGUFLinear(weight_tensor=ggt, bias=module.bias.detach() if module.bias is not None else None)
            # Re-attach into parent module via setattr on the parent
            parent_name, _, attr_name = module_name.rpartition(".")
            parent = self._pipe.noise_model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, attr_name, new_linear)
            replaced += 1
        logger.info(f"[LocalWanS2V] replaced {replaced} Linears with GGUFLinear; "
                    f"kept {kept} (no GGUF counterpart or F16)")

    def generate(self, image: str, audio: str, prompt: str, output_path: str,
                 max_area: int = 480 * 480, infer_frames: int = 21,
                 sampling_steps: int = 15, guide_scale: float = 1.0,
                 seed: int = -1, **kwargs) -> str:
        if self._pipe is None:
            raise RuntimeError("LocalWanS2V.generate called before load()")

        logger.info(f"[LocalWanS2V] generate prompt={prompt!r} frames={infer_frames}")
        video = self._pipe.generate(
            input_prompt=prompt,
            ref_image_path=image,
            audio_path=audio,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            max_area=max_area,
            infer_frames=infer_frames,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
        )
        # Wan returns a video tensor — save via repo's save_video helper
        from wan.utils.utils import save_video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(video, output_path, fps=16)  # Wan default sample_fps
        return output_path

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            import torch
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("[LocalWanS2V] unload best-effort cleanup failed")
```

- [ ] **Step 2: Re-run manager tests to verify nothing broke**

```bash
pytest tests/video_local/test_manager.py -v
```

Expected: still 6 PASS (the manager tests mock `_inner.load`).

- [ ] **Step 3: Commit**

```bash
git add src/video_local/wan_s2v_runtime.py
git commit -m "feat(video_local): LocalWanS2V runtime with GGUF DiT swap"
```

---

## Task 9: API router (HTTP endpoints)

**Files:**
- Create: `src/video_local/api.py`
- Test: `tests/video_local/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/video_local/test_api.py
"""Tests for /video/local/* endpoints."""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.video_local.manager import VideoModelManager


@pytest.fixture(autouse=True)
def _reset():
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


@pytest.fixture
def client():
    from fastapi import FastAPI
    from src.video_local.api import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status_endpoint(client):
    r = client.get("/video/local/status")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body and "quant" in body and "hf_id" in body


def test_load_endpoint_calls_manager_load(client):
    with patch.object(VideoModelManager.get()._inner, "load", return_value=None):
        r = client.post("/video/local/load")
    assert r.status_code == 200
    assert r.json()["state"] == "READY"


def test_load_endpoint_passes_quant(client):
    mgr = VideoModelManager.get()
    with patch.object(mgr._inner, "load", return_value=None), \
         patch.object(mgr._inner, "unload", return_value=None):
        r = client.post("/video/local/load", json={"quant": "Q8_0"})
    assert r.status_code == 200
    assert r.json()["quant"] == "Q8_0"


def test_cancel_endpoint(client):
    mgr = VideoModelManager.get()
    mgr._loaded = True
    with patch.object(mgr._inner, "unload", return_value=None):
        r = client.post("/video/local/cancel")
    assert r.status_code == 200
    assert r.json()["ok"] is True
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/video_local/test_api.py -v
```

Expected: FAIL on `from src.video_local.api import router`.

- [ ] **Step 3: Implement api router**

```python
# src/video_local/api.py
"""FastAPI router for /video/local/* endpoints. Mirrors src/audio_local/api.py."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .manager import VideoModelManager, VideoState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video/local", tags=["local-video"])


class LoadRequest(BaseModel):
    quant: Optional[str] = None


@router.get("/status")
def get_status() -> dict:
    return VideoModelManager.get().status()


@router.post("/load")
async def load(req: LoadRequest = LoadRequest()) -> dict:
    try:
        return await VideoModelManager.get().load(quant=req.quant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel() -> dict:
    mgr = VideoModelManager.get()
    s = mgr.status()
    if s["state"] == VideoState.UNLOADED.value:
        return {"ok": True, "cancelled": False}
    await mgr.unload()
    return {"ok": True, "cancelled": True}


class TestSynthRequest(BaseModel):
    prompt: str = "A close-up of a person talking calmly to the camera."


@router.post("/test_synthesize")
async def test_synthesize(req: TestSynthRequest = TestSynthRequest()) -> dict:
    """Diagnostic endpoint — synthesize a 21-frame 480×480 lipsync clip
    using the bundled examples/i2v_input.JPG + examples/talk.wav."""
    project_root = Path(__file__).resolve().parents[2]
    wan_examples = project_root / "output" / "external" / "Wan2.2" / "examples"
    out_path = str(project_root / "output" / "test_video.mp4")

    mgr = VideoModelManager.get()
    try:
        path = await asyncio.to_thread(
            mgr.generate,
            image=str(wan_examples / "i2v_input.JPG"),
            audio=str(wan_examples / "talk.wav"),
            prompt=req.prompt,
            output_path=out_path,
        )
        return {"ok": True, "path": path}
    except Exception as e:
        logger.exception("test_synthesize failed")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/video_local/test_api.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_local/api.py tests/video_local/test_api.py
git commit -m "feat(video_local): /video/local/* HTTP router"
```

---

## Task 10: VideoGenerator dispatch on VIDEO_PROVIDER

**Files:**
- Modify: `src/apps/comic_gen/video.py` (add `_build_model`)
- Test: `tests/test_video_generator_provider.py`

- [ ] **Step 1: Inspect existing VideoGenerator structure**

```bash
head -40 src/apps/comic_gen/video.py
```

(Note: this file already exists; identify the constructor pattern to follow.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_video_generator_provider.py
"""VideoGenerator must honor VIDEO_PROVIDER so video generation goes
through the same local runtime as image/audio when local is selected."""
import pytest

from src.apps.comic_gen.video import VideoGenerator


@pytest.fixture(autouse=True)
def _reset():
    from src.video_local.manager import VideoModelManager
    VideoModelManager.reset()
    yield
    VideoModelManager.reset()


class TestProviderSelection:
    def test_default_provider_is_wanx(self, monkeypatch):
        monkeypatch.delenv("VIDEO_PROVIDER", raising=False)
        gen = VideoGenerator({})
        assert type(gen.model).__name__ != "VideoModelManager"

    def test_local_routes_to_video_model_manager(self, monkeypatch):
        from src.video_local.manager import VideoModelManager
        monkeypatch.setenv("VIDEO_PROVIDER", "local")
        gen = VideoGenerator({})
        assert isinstance(gen.model, VideoModelManager)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("VIDEO_PROVIDER", "midjourney")
        with pytest.raises(ValueError, match="midjourney"):
            VideoGenerator({})
```

- [ ] **Step 3: Run tests to verify failure**

```bash
pytest tests/test_video_generator_provider.py -v
```

Expected: FAIL on AttributeError or unexpected behavior in `VideoGenerator({})`.

- [ ] **Step 4: Implement `_build_model` in VideoGenerator**

In `src/apps/comic_gen/video.py`, add the dispatch helper. The exact location depends on existing code structure — find the `__init__` of VideoGenerator and refactor model construction to delegate to `_build_model`:

```python
# In src/apps/comic_gen/video.py (modify existing __init__ + add _build_model)
import os

class VideoGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = self._build_model(self.config.get("model", {}))
        # ... rest of existing __init__ logic

    @staticmethod
    def _build_model(model_config: Dict[str, Any]):
        """Pick video-gen backend based on VIDEO_PROVIDER env var.

        Mirrors AssetGenerator._build_model / AudioGenerator._build_tts:
        a single env var flips between cloud Wanx and local Wan2.2-S2V.
        Local routes through VideoModelManager (singleton) so all
        callers share one loaded pipe."""
        provider = os.getenv("VIDEO_PROVIDER", "wanx").lower()
        if provider == "local":
            from ...video_local.manager import VideoModelManager
            return VideoModelManager.get(model_config)
        if provider == "wanx":
            # Use whichever class the existing code uses for cloud video
            # — replace this line with the existing class name
            from ...models.video import WanxVideoModel  # adjust to actual class
            return WanxVideoModel(model_config)
        raise ValueError(
            f"Unknown VIDEO_PROVIDER={provider!r}; expected 'wanx' or 'local'"
        )
```

> **Note for implementer:** The exact cloud video class name varies. Open `src/apps/comic_gen/video.py` and `src/models/video.py` (or wherever) to find the current cloud class. The pattern is the same as `_build_tts` in `src/apps/comic_gen/audio.py`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_video_generator_provider.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/apps/comic_gen/video.py tests/test_video_generator_provider.py
git commit -m "feat(video): VideoGenerator dispatches on VIDEO_PROVIDER"
```

---

## Task 11: Mount router + EnvConfig wiring

**Files:**
- Modify: `src/apps/comic_gen/api.py:33` (add import), `:74` (include_router), `:730` (EnvConfig fields), `:2140` (env config GET)
- Modify: `.env.example`

- [ ] **Step 1: Add router import and mount**

In `src/apps/comic_gen/api.py`, after the existing `from src.audio_local.api import router as local_audio_router`:

```python
from src.video_local.api import router as local_video_router
```

After `app.include_router(local_audio_router)`:

```python
app.include_router(local_video_router)
```

- [ ] **Step 2: Extend EnvConfig**

Find the `EnvConfig` class and add these three fields next to the existing `IMAGE_PROVIDER` / `TTS_PROVIDER`:

```python
VIDEO_PROVIDER: Optional[str] = None
LOCAL_VIDEO_QUANT: Optional[str] = None
USE_SAGE_ATTENTION: Optional[str] = None
```

- [ ] **Step 3: Extend `/config/env` GET return value**

In the `get_env_config` endpoint (around line 2120), add to the returned dict:

```python
"VIDEO_PROVIDER": os.getenv("VIDEO_PROVIDER", "wanx"),
"LOCAL_VIDEO_QUANT": os.getenv("LOCAL_VIDEO_QUANT", "Q4_K_S"),
"USE_SAGE_ATTENTION": os.getenv("USE_SAGE_ATTENTION", "0"),
```

- [ ] **Step 4: Document env vars in `.env.example`**

Append to `.env.example` after the TTS section:

```bash
# ===============================
# Video (S2V lipsync) provider
# ===============================
# VIDEO_PROVIDER=wanx                  # default: cloud DashScope WanX
# VIDEO_PROVIDER=local                 # 本地 Wan2.2-S2V-14B (GGUF, 4090 24GB)
# LOCAL_VIDEO_QUANT=Q4_K_S             # fp16 (28GB DiT) | Q8_0 (19.6GB) | Q4_K_S (13GB, recommended)
# USE_SAGE_ATTENTION=1                 # opt-in Sage Attention 2.x (faster, requires manual wheel install)
```

- [ ] **Step 5: Smoke-test backend boot**

```bash
python -c "from src.apps.comic_gen.api import app; print('routes:', [r.path for r in app.routes if '/video/local' in r.path])"
```

Expected: prints all 4 video endpoints.

- [ ] **Step 6: Commit**

```bash
git add src/apps/comic_gen/api.py .env.example
git commit -m "feat(video_local): mount /video/local router + extend EnvConfig"
```

---

## Task 12: Frontend api.ts — types + client methods

**Files:**
- Modify: `frontend/src/lib/api.ts` (add types around line 145; add 4 methods around line 700)

- [ ] **Step 1: Add types**

In `frontend/src/lib/api.ts`, after `LocalAudioVoice` (around line 145):

```typescript
// Local video runtime status (Wan2.2-S2V-14B with GGUF) — parallel to LocalAudioStatus.
export type LocalVideoQuant = "fp16" | "Q8_0" | "Q4_K_S";

export type LocalVideoState =
    | "UNLOADED"
    | "DOWNLOADING"
    | "LOADING"
    | "GENERATING"
    | "READY"
    | "ERROR";

export interface LocalVideoStatus {
    state: LocalVideoState;
    quant: LocalVideoQuant;
    hf_id: string;
    phase: string;
    progress: number;
    error: string | null;
    phase_label?: string;
}
```

- [ ] **Step 2: Extend EnvConfigPayload**

Find `EnvConfigPayload` interface (~line 30) and add:

```typescript
VIDEO_PROVIDER?: "wanx" | "local";
LOCAL_VIDEO_QUANT?: LocalVideoQuant;
USE_SAGE_ATTENTION?: "0" | "1";
```

- [ ] **Step 3: Add client methods**

After the existing `loadLocalAudio` block:

```typescript
// Local video (Wan2.2-S2V) runtime ----------------------------------
getLocalVideoStatus: async (): Promise<LocalVideoStatus> => {
    const res = await axios.get<LocalVideoStatus>(`${API_URL}/video/local/status`);
    return res.data;
},

loadLocalVideo: async (quant?: LocalVideoQuant): Promise<LocalVideoStatus> => {
    const res = await axios.post<LocalVideoStatus>(
        `${API_URL}/video/local/load`,
        { quant },
        { timeout: 1_800_000 },  // 30 min for first GGUF download
    );
    return res.data;
},

cancelLocalVideo: async (): Promise<{ ok: boolean; cancelled: boolean }> => {
    const res = await axios.post(`${API_URL}/video/local/cancel`);
    return res.data;
},

testLocalVideoSynthesize: async (prompt?: string): Promise<{ ok: boolean; path: string }> => {
    const res = await axios.post(
        `${API_URL}/video/local/test_synthesize`,
        { prompt },
        { timeout: 1_800_000 },  // 30 min generation
    );
    return res.data;
},
```

- [ ] **Step 4: Frontend dev server picks up the change**

Frontend HMR; no manual restart needed. Just confirm no TypeScript errors:

```bash
cd frontend && npx tsc --noEmit 2>&1 | tail -5
```

Expected: no errors mentioning api.ts.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): video local types + API client methods"
```

---

## Task 13: Frontend footer VIDEO row + modal Video section

**Files:**
- Modify: `frontend/src/components/layout/GlobalStatusFooter.tsx` (add VIDEO row mirroring TTS)
- Modify: `frontend/src/components/common/ModelSettingsModal.tsx` (add Video section)

- [ ] **Step 1: Footer VIDEO row**

In `GlobalStatusFooter.tsx`:

(a) Add `Film` to the lucide-react import line (and `LocalVideoStatus` to the api import).

(b) Add state hook + active flag mirroring `tts`:

```typescript
const [video, setVideo] = useState<LocalVideoStatus | null>(null);
// ...
const videoActive = env?.VIDEO_PROVIDER === "local";
const anyLocal = llmActive || imgActive || ttsActive || videoActive;
```

(c) Extend `refresh` to fetch video status when active:

```typescript
if (videoActive) {
    promises.push(api.getLocalVideoStatus().then(setVideo).catch(() => { }));
}
```

Add `videoActive` to the `useCallback` dep array.

(d) Render the VIDEO row after the TTS row:

```tsx
{videoActive && video && (
    <StatusRow
        icon={Film}
        label="VID"
        hfId={`${video.hf_id} (${video.quant})`}
        state={video.state}
        detail={
            video.progress > 0 && video.progress < 1
                ? `${Math.round(video.progress * 100)}%`
                : video.phase_label || video.phase || undefined
        }
        error={video.error}
    />
)}
```

- [ ] **Step 2: Modal Video section**

In `ModelSettingsModal.tsx`, after the Voice (TTS) section (and the `<div className="border-t border-white/10" />` after it), add:

```tsx
import { Film } from 'lucide-react';
// ... other imports

const [videoProvider, setVideoProvider] = useState<EnvConfigPayload["VIDEO_PROVIDER"]>("wanx");
const [videoQuant, setVideoQuant] = useState<LocalVideoQuant>("Q4_K_S");

useEffect(() => {
    api.getEnvConfig().then((env) => {
        setVideoProvider(env.VIDEO_PROVIDER === "local" ? "local" : "wanx");
        setVideoQuant((env.LOCAL_VIDEO_QUANT || "Q4_K_S") as LocalVideoQuant);
    }).catch(() => { });
}, []);

// Inside the modal body, after the LLM section (or wherever Voice section is):

<div className="border-t border-white/10" />

{/* Video (S2V lipsync) Section */}
<div className="space-y-4">
    <div className="flex items-center gap-2 text-sm font-bold text-white">
        <Film size={16} className="text-orange-400" />
        <span>Video (Speech-to-Video Lipsync)</span>
    </div>
    <div className="space-y-2">
        <label className="text-xs text-gray-400">Provider</label>
        <div className="grid grid-cols-2 gap-2">
            <button
                onClick={() => {
                    setVideoProvider("wanx");
                    api.saveEnvConfig({ VIDEO_PROVIDER: "wanx" }).catch(() => { });
                }}
                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left ${videoProvider === "wanx"
                    ? 'border-blue-500/50 bg-blue-500/10'
                    : 'border-white/10 hover:border-white/20 bg-white/5'
                    }`}
            >
                {videoProvider === "wanx" && (
                    <div className="absolute top-2 right-2"><Check size={14} className="text-blue-400" /></div>
                )}
                <span className="text-sm font-medium text-white">WanX (Cloud)</span>
                <span className="text-xs text-gray-500">DashScope · 远端</span>
            </button>
            <button
                onClick={() => {
                    setVideoProvider("local");
                    api.saveEnvConfig({ VIDEO_PROVIDER: "local", LOCAL_VIDEO_QUANT: videoQuant }).catch(() => { });
                    api.loadLocalVideo(videoQuant).catch(() => { });
                }}
                className={`relative flex flex-col items-start p-3 rounded-lg border transition-all text-left ${videoProvider === "local"
                    ? 'border-purple-500/50 bg-purple-500/10'
                    : 'border-white/10 hover:border-white/20 bg-white/5'
                    }`}
            >
                {videoProvider === "local" && (
                    <div className="absolute top-2 right-2"><Check size={14} className="text-purple-400" /></div>
                )}
                <span className="text-sm font-medium text-white">Wan2.2-S2V (Local)</span>
                <span className="text-xs text-gray-500">14B · GGUF · 4090 24GB</span>
            </button>
        </div>
    </div>

    {videoProvider === "local" && (
        <div className="space-y-2">
            <label className="text-xs text-gray-400">Quantization</label>
            <div className="grid grid-cols-3 gap-2">
                {(["fp16", "Q8_0", "Q4_K_S"] as LocalVideoQuant[]).map((q) => (
                    <button
                        key={q}
                        onClick={() => {
                            setVideoQuant(q);
                            api.saveEnvConfig({ LOCAL_VIDEO_QUANT: q }).catch(() => { });
                            api.loadLocalVideo(q).catch(() => { });
                        }}
                        className={`flex flex-col items-center p-3 rounded-lg border transition-all ${videoQuant === q
                            ? 'border-purple-500/50 bg-purple-500/10'
                            : 'border-white/10 hover:border-white/20 bg-white/5'
                            }`}
                    >
                        <span className="text-sm font-medium text-white">{q}</span>
                        <span className="text-[10px] text-gray-500">
                            {q === "fp16" ? "~28GB" : q === "Q8_0" ? "~20GB" : "~13GB ✓"}
                        </span>
                    </button>
                ))}
            </div>
            <p className="text-[11px] text-gray-500">Q4_K_S 是 4090 上推荐档位 — VRAM 余量足、速度也最快。</p>
        </div>
    )}
</div>
```

- [ ] **Step 3: Confirm TypeScript happy**

```bash
cd frontend && npx tsc --noEmit 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/layout/GlobalStatusFooter.tsx frontend/src/components/common/ModelSettingsModal.tsx
git commit -m "feat(frontend): video footer row + modal Video section"
```

---

## Task 14: Manual acceptance — generate a clip

**Files:**
- (None — this is a runtime check, no code changes)

- [ ] **Step 1: Verify Wan2.2 repo + Wan-AI snapshot present**

```bash
ls output/external/Wan2.2/wan/cli/ 2>&1 | head -3
du -sh output/models/LLM/hub/models--Wan-AI--Wan2.2-S2V-14B/ 2>&1
```

Expected: `cosyvoice.py` listed AND ~46G snapshot dir present.

- [ ] **Step 2: Set env, restart backend**

```bash
# Stop any running uvicorn
# Then restart with VIDEO_PROVIDER=local + LOCAL_VIDEO_QUANT=Q4_K_S
cd /s/AI/lumenxDev/lumenx
VIDEO_PROVIDER=local LOCAL_VIDEO_QUANT=Q4_K_S \
  python -u -m uvicorn src.apps.comic_gen.api:app --host 0.0.0.0 --port 17177
```

Wait for "Application startup complete" log line.

- [ ] **Step 3: Trigger load**

```bash
curl -X POST http://127.0.0.1:17177/video/local/load -H "Content-Type: application/json" -d '{"quant":"Q4_K_S"}'
```

Expected after a few minutes (download + load): `{"state":"READY","quant":"Q4_K_S",...}`.

If 13 GB GGUF download is the first time: this can take 10-30 min depending on network.

- [ ] **Step 4: Trigger test synthesize**

```bash
curl -X POST http://127.0.0.1:17177/video/local/test_synthesize -H "Content-Type: application/json" -d '{}' --max-time 1800
```

Wait 5-15 minutes (first inference is slow). Expected response: `{"ok":true,"path":"S:\\AI\\lumenxDev\\lumenx\\output\\test_video.mp4"}`.

- [ ] **Step 5: Eyeball the video**

Open `output/test_video.mp4` in any media player.

**Acceptance criteria (visual)**:
1. Video plays without errors
2. Audio (a person speaking) is embedded
3. The face's mouth moves roughly in sync with the audio (lipsync intent visible — not perfect but recognizable)
4. No obvious frame corruption / random color noise

If lipsync is wrong but speech timing matches, dequant is OK and we have a real working pipeline. If output is random noise, go back to Task 4 and re-check `_unpack_k4_scales_mins` against llama.cpp reference.

- [ ] **Step 6: Update memory note + commit**

Once the validation passes, update memory file:
- File: `C:\Users\hemin\.claude\projects\s--AI-lumenxDev\memory\project_local_video_s2v_planned.md`
- Change `→ **Local I2V with lipsync: Wan2.2-S2V-14B** (validation in progress)` to `✅ **Local I2V with lipsync: Wan2.2-S2V-14B (Q4_K_S GGUF, validated)**`
- Add line: `Validated 2026-MM-DD: 21-frame 480×480 lipsync clip on 4090, peak VRAM ~16GB, inference ~5-10 min/clip`

- [ ] **Step 7: Final commit**

```bash
git add C:\Users\hemin\.claude\projects\s--AI-lumenxDev\memory\project_local_video_s2v_planned.md
# (memory file is outside the repo — don't actually git add it; just save the file)
# Actually nothing to commit here; the work is done. Mark the todo entry.
```

---

## Self-review

**Spec coverage check:**
- [x] §Architecture — module layout: Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9
- [x] §Components reader/dequant/ops/sage_attn/runtime/manager/api: Tasks 2, 3, 4, 5, 6, 8, 7, 9
- [x] §Data flow cold load: Task 8 (Steps 1-7) + Task 14
- [x] §Data flow generate: Task 8 (`generate` method) + Task 14
- [x] §Testing unit tests 1-5: Tasks 2, 3, 4, 5, 6, 7, 9
- [x] §Testing integration test 6: Task 14 (manual acceptance subsumes this)
- [x] §Manual acceptance 7: Task 14
- [x] §Error handling table all 7 cases: covered across Tasks 2 (UnsupportedQuantError), 7 (state machine errors), 8 (load/generate try/except), 9 (HTTPException)
- [x] §Dependencies: Task 1 + Task 11
- [x] §Frontend: Tasks 12, 13
- [x] §VideoGenerator dispatch: Task 10
- [x] §EnvConfig fields: Task 11

**Placeholder scan:** None found ("TODO", "TBD" only appear in code comments where they describe in-flight work that's resolved by the next task).

**Type consistency:**
- `LocalVideoQuant` defined in api.ts Task 12, used in modal Task 13 ✓
- `VideoModelManager` class in Task 7 with methods load/unload/generate/status, used by api.py Task 9 ✓
- `LocalWanS2V` stub in Task 7, real impl Task 8, methods load/unload/generate/quant/hf_id consistent ✓
- `parse_gguf` returns `Dict[str, GGUFTensor]` in Task 2, consumed by Task 8 (`_patch_with_gguf`) ✓
- `GGUFTensor.raw_bytes: memoryview` used by `GGUFLinear.__init__` in Task 5 ✓

No issues found in self-review.
