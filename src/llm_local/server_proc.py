"""LlamaServerProcess — subprocess wrapper for llama-server.exe."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class ServerStartTimeout(RuntimeError):
    """Raised when llama-server's /health didn't return 200 within startup_timeout."""


class LlamaServerProcess:
    """Owns one llama-server.exe subprocess.

    - start() spawns the process and polls /health until ready (or timeout).
    - terminate() sends SIGTERM, falls back to SIGKILL after grace period.
    - is_alive() reflects current process state.
    """

    def __init__(
        self,
        exe: Path,
        gguf_path: Path,
        port: int = 17178,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        startup_poll_interval: float = 0.5,
        extra_args: Optional[List[str]] = None,
    ):
        self.exe = Path(exe)
        self.gguf_path = Path(gguf_path)
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.startup_poll_interval = startup_poll_interval
        self.extra_args = extra_args or []
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _build_cmd(self) -> List[str]:
        # KV cache quantized to q8_0 (~halves attention KV vs f16) — needed to
        # fit a 35B-A3B model + 65k ctx on a 24GB card. V-cache quant requires
        # --flash-attn or llama-server rejects it.
        return [
            str(self.exe),
            "-m", str(self.gguf_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.n_ctx),
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            # Single shared KV buffer (LM Studio's "Unified KV Cache"). For our
            # single-sequence usage this avoids per-sequence KV slot reservation.
            "--kv-unified",
            *self.extra_args,
        ]

    def start(self, startup_timeout: float = 60.0) -> None:
        """Spawn llama-server.exe and block until /health returns 200, or raise.

        Raises:
            ServerStartTimeout: if /health didn't go healthy within startup_timeout
            RuntimeError: if the process exited before becoming healthy
        """
        cmd = self._build_cmd()
        logger.info(f"Spawning: {' '.join(cmd)}")

        # Capture stderr so we can show it on startup failure
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"llama-server pid={self._proc.pid}")

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            # Did the process die?
            ret = self._proc.poll()
            if ret is not None:
                # Read stderr tail for diagnostics
                try:
                    err = (self._proc.stderr.read() or b"").decode("utf-8", errors="replace")
                except Exception:
                    err = ""
                self._proc = None
                raise RuntimeError(
                    f"llama-server exited before becoming healthy "
                    f"(returncode={ret}). stderr tail:\n{err[-1500:]}"
                )

            # Try /health
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    logger.info(f"llama-server healthy on {self.base_url}")
                    return
            except Exception:
                pass  # not ready yet, keep polling

            time.sleep(self.startup_poll_interval)

        # Timeout
        self.terminate(grace_seconds=2.0)
        raise ServerStartTimeout(
            f"llama-server /health did not return 200 within {startup_timeout}s"
        )

    def terminate(self, grace_seconds: float = 5.0) -> None:
        """Send SIGTERM and wait up to grace_seconds, then SIGKILL if still alive.

        Idempotent: safe to call when no process is tracked.
        """
        if self._proc is None:
            return
        p = self._proc
        try:
            p.terminate()
            try:
                p.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                logger.warning(f"llama-server pid={p.pid} did not exit in {grace_seconds}s; killing")
                p.kill()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.error(f"llama-server pid={p.pid} is unkillable?!")
        finally:
            self._proc = None
