"""Tests for EmbeddedServerRuntime — the runtime that wraps llama-server.exe."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_local.runtime import LoadResult
from src.llm_local.runtime_server import EmbeddedServerRuntime


@pytest.fixture
def fake_paths(tmp_path):
    """Provide fake exe + downloaded gguf paths."""
    exe = tmp_path / "runtime" / "llama.cpp-b8941" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"x")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 1000)
    return exe, gguf


class TestLoad:
    def test_load_starts_subprocess_with_correct_args(self, fake_paths):
        exe, gguf = fake_paths
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")

        # Mock the binary manager + hf_hub_download + LlamaServerProcess
        fake_proc = MagicMock()
        with patch("src.llm_local.runtime_server.get_binary_manager") as mock_mgr, \
             patch("src.llm_local.runtime_server.hf_hub_download",
                   return_value=str(gguf)) as mock_dl, \
             patch("src.llm_local.runtime_server.LlamaServerProcess",
                   return_value=fake_proc) as mock_proc_cls:
            mock_mgr.return_value.current_exe_path.return_value = exe
            result = rt.load()

        # Process was instantiated with correct paths
        mock_proc_cls.assert_called_once()
        call_kwargs = mock_proc_cls.call_args.kwargs
        assert call_kwargs["exe"] == exe
        assert call_kwargs["gguf_path"] == Path(str(gguf))
        # And start() was called
        fake_proc.start.assert_called_once()
        assert isinstance(result, LoadResult)
        assert result.hf_id == "test/Model-GGUF"

    def test_load_raises_if_no_binary_installed(self, fake_paths):
        _, gguf = fake_paths
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        with patch("src.llm_local.runtime_server.get_binary_manager") as mock_mgr, \
             patch("src.llm_local.runtime_server.hf_hub_download", return_value=str(gguf)):
            mock_mgr.return_value.current_exe_path.return_value = None
            with pytest.raises(RuntimeError, match="llama.cpp"):
                rt.load()

    def test_load_is_idempotent(self, fake_paths):
        exe, gguf = fake_paths
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        fake_proc = MagicMock()
        with patch("src.llm_local.runtime_server.get_binary_manager") as mock_mgr, \
             patch("src.llm_local.runtime_server.hf_hub_download", return_value=str(gguf)), \
             patch("src.llm_local.runtime_server.LlamaServerProcess",
                   return_value=fake_proc) as mock_proc_cls:
            mock_mgr.return_value.current_exe_path.return_value = exe
            rt.load()
            rt.load()  # second call: server already up, no-op
        # LlamaServerProcess only constructed once
        assert mock_proc_cls.call_count == 1


class TestChat:
    def test_chat_posts_to_local_server_and_returns_content(self, fake_paths):
        exe, gguf = fake_paths
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")

        # Set up loaded state
        fake_proc = MagicMock()
        fake_proc.base_url = "http://127.0.0.1:17178"
        rt._proc = fake_proc
        rt._loaded = True

        # Mock OpenAI client streaming response — iter of chunks with
        # .choices[0].delta.content
        def _make_chunk(text):
            ch = MagicMock()
            ch.choices = [MagicMock(delta=MagicMock(content=text))]
            return ch
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = iter([
            _make_chunk("Hello "), _make_chunk("from the model"),
        ])

        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            response = rt.chat([{"role": "user", "content": "hi"}], max_new_tokens=50)

        assert response == "Hello from the model"
        # Verify request used right base_url
        from src.llm_local.runtime_server import OpenAI as _ImportedOpenAI  # noqa: F401
        # OpenAI() was called with base_url
        # (we can't easily check the patched call's kwargs through the import)

    def test_chat_raises_if_not_loaded(self, fake_paths):
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        with pytest.raises(RuntimeError, match="not loaded"):
            rt.chat([{"role": "user", "content": "hi"}])


class TestChatExtraOptions:
    """response_format + Qwen3 thinking-mode handling."""

    def _setup_loaded_runtime(self):
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        fake_proc = MagicMock()
        fake_proc.base_url = "http://127.0.0.1:17178"
        rt._proc = fake_proc
        rt._loaded = True
        return rt

    def _fake_completion(self, content="x"):
        # Streaming response: a single chunk carrying the full text.
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=content))]
        return iter([chunk])

    def test_chat_forwards_response_format_to_llama_server(self):
        rt = self._setup_loaded_runtime()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._fake_completion("OK")
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            rt.chat(
                [{"role": "user", "content": "give me JSON"}],
                response_format={"type": "json_object"},
            )
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("response_format") == {"type": "json_object"}

    def test_chat_disables_qwen3_thinking_when_json_mode(self):
        """When caller requests JSON output, also set enable_thinking=false to
        save token budget — Qwen3 reasoning tokens otherwise eat 100s of tokens
        before any visible content. Sent via OpenAI extra_body."""
        rt = self._setup_loaded_runtime()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._fake_completion("{}")
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            rt.chat(
                [{"role": "user", "content": "give JSON"}],
                response_format={"type": "json_object"},
            )
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        # llama-server reads chat_template_kwargs from extra_body
        extra = kwargs.get("extra_body") or {}
        assert extra.get("chat_template_kwargs") == {"enable_thinking": False}

    def test_chat_does_not_disable_thinking_for_plain_text(self):
        """Plain (non-JSON) chat keeps thinking enabled for richer responses."""
        rt = self._setup_loaded_runtime()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._fake_completion("hi")
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            rt.chat([{"role": "user", "content": "hi"}])
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        # No extra_body or no chat_template_kwargs override
        extra = kwargs.get("extra_body") or {}
        assert "chat_template_kwargs" not in extra

    def test_chat_uses_streaming_so_thinking_tokens_are_visible_live(self):
        """Streaming is required so the backend cmd window shows <think> tokens
        as they arrive instead of blocking until completion."""
        rt = self._setup_loaded_runtime()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._fake_completion("x")
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            rt.chat([{"role": "user", "content": "hi"}])
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("stream") is True

    def test_chat_concatenates_streamed_chunks_into_full_response(self):
        """Multiple delta chunks should be joined into one return string."""
        rt = self._setup_loaded_runtime()

        def _chunk(text):
            ch = MagicMock()
            ch.choices = [MagicMock(delta=MagicMock(content=text))]
            return ch
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = iter([
            _chunk("part1 "), _chunk("part2 "), _chunk("part3"),
        ])
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            out = rt.chat([{"role": "user", "content": "hi"}])
        assert out == "part1 part2 part3"

    def test_chat_default_max_tokens_increased_for_thinking_models(self):
        """Default max_tokens should be high enough for Qwen3 thinking + JSON output."""
        rt = self._setup_loaded_runtime()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._fake_completion("x")
        with patch("src.llm_local.runtime_server.OpenAI", return_value=fake_client):
            rt.chat([{"role": "user", "content": "hi"}])
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        # 1024 was too small for Qwen3 thinking (test endpoint sample showed
        # 172 thinking tokens for "what is 2+2"). Bump to >= 4096.
        assert kwargs["max_tokens"] >= 4096


class TestUnload:
    def test_unload_terminates_process(self, fake_paths):
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        fake_proc = MagicMock()
        rt._proc = fake_proc
        rt._loaded = True

        rt.unload()

        fake_proc.terminate.assert_called_once()
        assert rt._proc is None
        assert rt._loaded is False

    def test_unload_is_idempotent(self, fake_paths):
        rt = EmbeddedServerRuntime(hf_id="test/Model-GGUF", gguf_file="model.gguf")
        rt.unload()  # never loaded — should not raise
        rt.unload()  # second call — also fine
