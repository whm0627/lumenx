"""Tests for src/video_local/gguf/reader.py.

We don't ship a sample .gguf in the repo (sizes range from MB to GB).
Instead each test creates a tiny GGUF on the fly using the upstream
`gguf` package's writer, then parses it with our reader and verifies
the round-trip."""
from pathlib import Path

import numpy as np
import pytest

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
        monkeypatch.setattr(r, "_gguf_type_to_name", lambda t: "Q3_K_M")
        with pytest.raises(UnsupportedQuantError, match="Q3_K_M"):
            parse_gguf(str(f))
