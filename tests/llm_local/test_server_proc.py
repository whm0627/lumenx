"""Tests for LlamaServerProcess — subprocess wrapper for llama-server.exe."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_local.server_proc import LlamaServerProcess, ServerStartTimeout


def make_proc_mock(returncode=None, alive=True):
    """A subprocess.Popen-like mock."""
    p = MagicMock()
    # poll() returns None if alive, else returncode
    p.poll.return_value = None if alive else returncode
    p.returncode = returncode
    p.pid = 12345
    return p


def make_health_response(ok: bool):
    """Mock httpx.get response."""
    r = MagicMock()
    r.status_code = 200 if ok else 503
    return r


class TestStart:
    def test_start_polls_health_until_ok(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf, port=17178,
                                  startup_poll_interval=0.01)

        # First 2 health calls fail (server not ready), then succeed
        responses = [Exception("connection refused"),
                     Exception("connection refused"),
                     make_health_response(ok=True)]

        with patch("src.llm_local.server_proc.subprocess.Popen",
                   return_value=make_proc_mock()) as mock_popen, \
             patch("src.llm_local.server_proc.httpx.get",
                   side_effect=responses) as mock_get:
            proc.start(startup_timeout=5.0)

        mock_popen.assert_called_once()
        # Args: exe + -m + gguf + --port + 17178 (other defaults too)
        cmd = mock_popen.call_args[0][0]
        assert str(exe) in cmd[0] or cmd[0] == str(exe)
        assert "-m" in cmd
        assert str(gguf) in cmd
        assert "--port" in cmd
        assert "17178" in cmd

        assert mock_get.call_count == 3
        assert proc.is_alive() is True

    def test_start_raises_on_timeout(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf, port=17178,
                                  startup_poll_interval=0.01)

        with patch("src.llm_local.server_proc.subprocess.Popen",
                   return_value=make_proc_mock()), \
             patch("src.llm_local.server_proc.httpx.get",
                   side_effect=Exception("connection refused")):
            with pytest.raises(ServerStartTimeout):
                proc.start(startup_timeout=0.1)

    def test_start_raises_if_process_dies_during_startup(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf, port=17178,
                                  startup_poll_interval=0.01)

        # Process dies (poll returns non-None) before health check succeeds
        dead_proc = make_proc_mock(returncode=1, alive=False)
        dead_proc.stderr = MagicMock()
        dead_proc.stderr.read.return_value = b"CUDA error\n"

        with patch("src.llm_local.server_proc.subprocess.Popen",
                   return_value=dead_proc), \
             patch("src.llm_local.server_proc.httpx.get",
                   side_effect=Exception("connection refused")):
            with pytest.raises(RuntimeError, match="exited.*1"):
                proc.start(startup_timeout=5.0)


class TestTerminate:
    def test_terminate_calls_terminate_then_waits(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        mock_p = make_proc_mock()
        proc._proc = mock_p

        # wait() returns 0 (clean exit) within grace
        mock_p.wait.return_value = 0

        proc.terminate(grace_seconds=1.0)

        mock_p.terminate.assert_called_once()
        mock_p.wait.assert_called_once_with(timeout=1.0)
        # Did NOT need to kill
        mock_p.kill.assert_not_called()
        assert proc._proc is None

    def test_terminate_kills_after_grace_timeout(self, tmp_path):
        import subprocess as _sub
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        mock_p = make_proc_mock()
        proc._proc = mock_p

        # First wait() raises TimeoutExpired (process didn't exit), kill, second wait succeeds
        mock_p.wait.side_effect = [_sub.TimeoutExpired("cmd", 1.0), 0]

        proc.terminate(grace_seconds=1.0)

        mock_p.terminate.assert_called_once()
        mock_p.kill.assert_called_once()
        assert mock_p.wait.call_count == 2

    def test_terminate_is_idempotent(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        # _proc is None — never started, or already terminated
        proc.terminate()  # should not raise
        proc.terminate()  # second call also fine


class TestIsAlive:
    def test_returns_false_before_start(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        assert proc.is_alive() is False

    def test_returns_true_when_process_running(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        proc._proc = make_proc_mock(alive=True)
        assert proc.is_alive() is True

    def test_returns_false_when_process_exited(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        proc._proc = make_proc_mock(returncode=0, alive=False)
        assert proc.is_alive() is False


class TestBuildCmd:
    """Verify VRAM-saving flags are wired into the spawn cmd."""

    def test_build_cmd_includes_flash_attn_and_q8_kv_cache(self, tmp_path):
        # KV cache q8_0 ~halves attention KV vs f16 (4-6 GB on 65k ctx 35B);
        # V-cache quant requires --flash-attn so all three must travel together.
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf)
        cmd = proc._build_cmd()
        # llama.cpp b8941+ requires --flash-attn to take a value (on|off|auto)
        assert "--flash-attn" in cmd
        assert cmd[cmd.index("--flash-attn") + 1] == "on"
        assert "--cache-type-k" in cmd
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
        assert "--cache-type-v" in cmd
        assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0"
        assert "--kv-unified" in cmd


class TestBaseUrl:
    def test_base_url_uses_configured_port(self, tmp_path):
        exe = tmp_path / "llama-server.exe"
        exe.write_bytes(b"x")
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        proc = LlamaServerProcess(exe=exe, gguf_path=gguf, port=12345)
        assert proc.base_url == "http://127.0.0.1:12345"
