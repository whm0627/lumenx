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
