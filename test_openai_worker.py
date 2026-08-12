#!/usr/bin/env python3
"""Test OpenAIWorker integration with KiroCrew knowledge LLMPool.

Run::

    cd openai_provider
    python -m unittest test_openai_worker -v
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure openai_provider is importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestSharedConfig(unittest.TestCase):
    """Tests for _config.py shared configuration."""

    def test_lookup_context_window_exact(self):
        from openai_provider._config import lookup_context_window
        self.assertEqual(lookup_context_window("gpt-4o"), 128_000)
        self.assertEqual(lookup_context_window("free"), 1_048_576)

    def test_lookup_context_window_prefix(self):
        from openai_provider._config import lookup_context_window
        self.assertEqual(lookup_context_window("gpt-4o-custom"), 128_000)

    def test_lookup_context_window_unknown(self):
        from openai_provider._config import lookup_context_window
        self.assertIsNone(lookup_context_window("unknown-model-xyz"))

    def test_read_provider_config(self):
        from openai_provider._config import read_provider_config
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "http://custom:8080/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "my-model",
            "OPENAI_MAX_TOKENS": "4096",
            "OPENAI_CONTEXT_WINDOW": "128000",
            "OPENAI_SYSTEM_PROMPT": "test prompt",
        }):
            cfg = read_provider_config()
            self.assertEqual(cfg["base_url"], "http://custom:8080/v1")
            self.assertEqual(cfg["api_key"], "sk-test")
            self.assertEqual(cfg["model"], "my-model")
            self.assertEqual(cfg["max_tokens"], 4096)
            self.assertEqual(cfg["context_window"], 128000)
            self.assertEqual(cfg["system_prompt"], "test prompt")

    def test_read_knowledge_config_prefers_knowledge_model(self):
        from openai_provider._config import read_knowledge_config
        with patch.dict(os.environ, {
            "OPENAI_MODEL": "main-model",
            "OPENAI_KNOWLEDGE_MODEL": "knowledge-model",
            "OPENAI_API_KEY": "sk-test",
        }):
            cfg = read_knowledge_config()
            self.assertEqual(cfg["model"], "knowledge-model")

    def test_read_knowledge_config_falls_back_to_openai_model(self):
        from openai_provider._config import read_knowledge_config
        with patch.dict(os.environ, {
            "OPENAI_MODEL": "fallback-model",
            "OPENAI_API_KEY": "sk-test",
        }, clear=False):
            os.environ.pop("OPENAI_KNOWLEDGE_MODEL", None)
            cfg = read_knowledge_config()
            self.assertEqual(cfg["model"], "fallback-model")


class TestOpenAIWorkerUnit(unittest.TestCase):
    """Unit tests for OpenAIWorker without real API calls."""

    def test_import(self):
        from openai_provider.openai_worker import OpenAIWorker
        self.assertTrue(callable(OpenAIWorker))

    def test_monkey_patch_applies(self):
        from openai_provider.openai_worker import _register_openai_worker
        from kiro_crew.knowledge.llm_pool import LLMPool

        orig = LLMPool._create_worker
        _register_openai_worker()
        self.assertIsNot(LLMPool._create_worker, orig)
        LLMPool._create_worker = orig

    def test_patch_returns_openai_worker(self):
        asyncio.run(self._test_patch_returns_openai_worker())

    async def _test_patch_returns_openai_worker(self):
        from openai_provider.openai_worker import _register_openai_worker, OpenAIWorker
        from kiro_crew.knowledge.llm_pool import LLMPool

        orig = LLMPool._create_worker
        _register_openai_worker()

        pool = LLMPool(pool_size=1)
        pool._provider_type = "openai"

        with patch.object(OpenAIWorker, "start", new_callable=AsyncMock):
            worker = await pool._create_worker()
            self.assertIsInstance(worker, OpenAIWorker)

        LLMPool._create_worker = orig

    def test_patch_falls_back_to_original(self):
        asyncio.run(self._test_patch_falls_back())

    async def _test_patch_falls_back(self):
        from openai_provider.openai_worker import _register_openai_worker
        from kiro_crew.knowledge.llm_pool import LLMPool

        orig = LLMPool._create_worker
        _register_openai_worker()

        pool = LLMPool(pool_size=1)
        pool._provider_type = "acp"
        pool._sandbox_mode = "off"

        with patch("kiro_crew.knowledge.llm_pool.AcpWorker") as MockAcp:
            mock_worker = AsyncMock()
            mock_worker.is_alive.return_value = True
            MockAcp.return_value = mock_worker
            worker = await pool._create_worker()
            self.assertIs(worker, mock_worker)

        LLMPool._create_worker = orig

    def test_install_registers_worker(self):
        import importlib
        _mod = importlib.import_module("openai_provider.install")
        _mod._installed = False

        from kiro_crew.knowledge.llm_pool import LLMPool
        orig = LLMPool._create_worker

        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "http://test:1234/v1",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
        }):
            _mod.install()

        self.assertIsNot(LLMPool._create_worker, orig)
        LLMPool._create_worker = orig


class TestOpenAIWorkerAsync(unittest.TestCase):
    """Async tests for OpenAIWorker with mocked HTTP."""

    def test_lifecycle(self):
        asyncio.run(self._test_lifecycle())

    async def _test_lifecycle(self):
        from openai_provider.openai_worker import OpenAIWorker

        worker = OpenAIWorker()

        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "http://test:1234/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "test-model",
        }):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '{"title": "test"}',
                    }
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()

            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {"data": []}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.get = AsyncMock(return_value=mock_models_response)
            mock_client.aclose = AsyncMock()

            with patch("openai_provider.openai_worker._httpx") as mock_httpx:
                mock_httpx.AsyncClient.return_value = mock_client
                mock_httpx.Timeout = MagicMock()

                await worker.start()
                self.assertTrue(worker.is_alive())

                result = await worker.send_message("Test prompt")
                self.assertIn("title", result)

                await worker.reset_conversation()
                self.assertEqual(worker.calls_since_reset, 0)

                await worker.shutdown()
                self.assertFalse(worker.is_alive())

    def test_context_pct(self):
        from openai_provider.openai_worker import OpenAIWorker
        worker = OpenAIWorker()
        worker._context_window = 100_000
        worker._prompt_tokens = 50_000
        self.assertAlmostEqual(worker.context_pct(), 50.0)

        worker._context_window = 0
        self.assertEqual(worker.context_pct(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
