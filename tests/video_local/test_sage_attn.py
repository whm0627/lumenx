"""Tests for the Sage Attention monkeypatch shim. We don't actually
require the sageattention package to be installed — the shim must
fall through silently if it's missing."""
import sys

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
    assert callable(F.scaled_dot_product_attention)


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
