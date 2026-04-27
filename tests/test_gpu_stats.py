"""Tests for read_gpu_stats — thin wrapper around nvidia-smi.

The footer polls this every couple of seconds, so it must:
- parse nvidia-smi's CSV cleanly when CUDA is present
- degrade gracefully (zeros) when nvidia-smi is missing or errors
- never raise (it's a UI status read, not a critical path)
"""
from unittest.mock import MagicMock, patch

from src.utils.gpu_stats import read_gpu_stats


class TestParseHappyPath:
    def test_parses_used_and_total_from_csv(self):
        # nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
        # → "  18432, 24563"
        completed = MagicMock(returncode=0, stdout="18432, 24563\n")
        with patch("subprocess.run", return_value=completed):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 18432, "total_mb": 24563}

    def test_handles_extra_whitespace(self):
        completed = MagicMock(returncode=0, stdout="  100,  500  \n")
        with patch("subprocess.run", return_value=completed):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 100, "total_mb": 500}


class TestFallback:
    def test_returns_zeros_when_subprocess_raises_filenotfound(self):
        # nvidia-smi not on PATH — common on non-NVIDIA machines / CI
        with patch("subprocess.run", side_effect=FileNotFoundError):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 0, "total_mb": 0}

    def test_returns_zeros_when_subprocess_returns_nonzero(self):
        completed = MagicMock(returncode=1, stdout="", stderr="some error")
        with patch("subprocess.run", return_value=completed):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 0, "total_mb": 0}

    def test_returns_zeros_when_output_unparseable(self):
        completed = MagicMock(returncode=0, stdout="garbage data\n")
        with patch("subprocess.run", return_value=completed):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 0, "total_mb": 0}

    def test_returns_zeros_when_subprocess_raises_generic(self):
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            stats = read_gpu_stats()
        assert stats == {"used_mb": 0, "total_mb": 0}
