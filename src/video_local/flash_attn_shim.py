"""SDPA-based shim for `wan.modules.attention.flash_attention`.

Wan-S2V calls `flash_attention()` directly (not the `attention()`
dispatcher with its SDPA fallback) — see model_s2v.py:171. When the
flash_attn package is missing, that direct call asserts on
`FLASH_ATTN_2_AVAILABLE` and the run dies.

flash-attn doesn't ship a Windows wheel for torch 2.11+cu128 today; a
from-source build with our CUDA 13.2 has a non-trivial chance of
failing. The pragmatic fix: replace the `flash_attention` symbol in
every Wan module that imported it at top level with a SDPA wrapper.

Math note: `F.scaled_dot_product_attention` over fully-attended
[B, N, L, C] is equivalent to FA2's varlen kernel for batch=1,
q_lens=None, full-length k_lens — which is exactly Wan-S2V's
inference-time call signature. For the rare uneven-k_lens case we
build an additive attention mask.

Patch targets: every Wan module that did `from ..attention import
flash_attention` or `from ..model import flash_attention`. Patching
just `wan.modules.attention.flash_attention` is NOT enough — those
imports captured the function object by reference at module-load time,
so the caller's globals dict still points at the original.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def sdpa_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version=None,
) -> torch.Tensor:
    """Match Wan's flash_attention signature.

    Inputs:  q,k,v shape [B, L, N, C] (sequence-major).
    Output:  shape [B, Lq, N, C2] (sequence-major), dtype = q.dtype.
    """
    out_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    if q.dtype not in half_dtypes:
        q = q.to(dtype)
    if k.dtype not in half_dtypes:
        k = k.to(dtype)
    if v.dtype not in half_dtypes:
        v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    needs_mask = False
    if q_lens is not None:
        Lq = q.size(1)
        if not torch.equal(q_lens, torch.full_like(q_lens, Lq)):
            needs_mask = True
    if k_lens is not None:
        Lk = k.size(1)
        if not torch.equal(k_lens, torch.full_like(k_lens, Lk)):
            needs_mask = True

    # [B, L, N, C] -> [B, N, L, C] for SDPA's head-major layout.
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2).to(q_h.dtype)
    v_h = v.transpose(1, 2).to(q_h.dtype)

    attn_mask = None
    if needs_mask and k_lens is not None:
        B, _, Lk, _ = k_h.shape
        idx = torch.arange(Lk, device=k_h.device).unsqueeze(0)         # [1, Lk]
        valid = idx < k_lens.to(k_h.device).unsqueeze(1)               # [B, Lk]
        attn_mask = valid.unsqueeze(1).unsqueeze(2)                    # [B, 1, 1, Lk]

    out = F.scaled_dot_product_attention(
        q_h, k_h, v_h,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    # Back to [B, L, N, C]
    out = out.transpose(1, 2).contiguous()
    return out.to(out_dtype)


_PATCH_TARGETS = (
    "wan.modules.attention",
    "wan.modules.model",          # WanAttentionBlock parent class —
                                  # easy to miss because it's reached
                                  # via inheritance from
                                  # WanS2VAttentionBlock.
    "wan.modules.s2v.model_s2v",
    "wan.modules.animate.clip",
    "wan.modules.animate.face_blocks",
    "wan.distributed.ulysses",
)


def patch_wan_flash_attention() -> int:
    """Replace `flash_attention` symbol in every Wan module that
    imported it at top level. Returns the number of modules patched.
    Call this AFTER WanS2V has been constructed (so all the modules are
    already imported into sys.modules)."""
    import sys
    patched = []
    for mod_name in _PATCH_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if not hasattr(mod, "flash_attention"):
            continue
        setattr(mod, "flash_attention", sdpa_flash_attention)
        patched.append(mod_name)
    if patched:
        logger.info(
            "[flash_attn_shim] patched flash_attention -> SDPA in %d Wan module(s): %s",
            len(patched), patched,
        )
    else:
        logger.warning(
            "[flash_attn_shim] no Wan modules imported yet — call this AFTER "
            "WanS2V is constructed (or it'll be a no-op)"
        )
    return len(patched)
