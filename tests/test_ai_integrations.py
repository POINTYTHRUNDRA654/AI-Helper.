"""Tests for ai_helper.ai_integrations."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from ai_helper.ai_integrations import (
    AIAppRegistry,
    AIAppStatus,
    GenerateResult,
    OllamaClient,
    OllamaModel,
    _get,
    _post,
    _reachable,
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class TestHttpHelpers(unittest.TestCase):
    def test_get_returns_none_on_error(self):
        with patch("ai_helper.ai_integrations.urllib.request.urlopen", side_effect=OSError):
            result = _get("http://localhost:9999/does-not-exist")
        self.assertIsNone(result)

    def test_get_parses_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"key": "value"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("ai_helper.ai_integrations.urllib.request.urlopen", return_value=mock_resp):
            result = _get("http://localhost:11434/api/tags")
        self.assertEqual(result, {"key": "value"})

    def test_post_returns_none_on_error(self):
        with patch("ai_helper.ai_integrations.urllib.request.urlopen", side_effect=OSError):
            result = _post("http://localhost:9999/", {})
        self.assertIsNone(result)

    def test_reachable_true_on_http_error(self):
        import urllib.error
        with patch(
            "ai_helper.ai_integrations.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None),
        ):
            self.assertTrue(_reachable("http://localhost:11434"))

    def test_reachable_false_on_connection_error(self):
        with patch("ai_helper.ai_integrations.urllib.request.urlopen", side_effect=OSError):
            self.assertFalse(_reachable("http://localhost:9999"))


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(base_url="http://localhost:11434")

    def _mock_get(self, data):
        return patch("ai_helper.ai_integrations._get", return_value=data)

    def _mock_post(self, data):
        return patch("ai_helper.ai_integrations._post", return_value=data)

    def _mock_reachable(self, value):
        return patch("ai_helper.ai_integrations._reachable", return_value=value)

    def test_is_running_true(self):
        with self._mock_reachable(True):
            self.assertTrue(self.client.is_running())

    def test_is_running_false(self):
        with self._mock_reachable(False):
            self.assertFalse(self.client.is_running())

    def test_status_running(self):
        with self._mock_reachable(True), \
             self._mock_get({"version": "0.1.14"}):
            status = self.client.status()
        self.assertTrue(status.running)
        self.assertEqual(status.name, "Ollama")

    def test_status_not_running(self):
        with self._mock_reachable(False), \
             self._mock_get(None):
            status = self.client.status()
        self.assertFalse(status.running)

    def test_list_models(self):
        tags_data = {
            "models": [
                {"name": "llama3:latest", "size": 4_000_000_000, "modified_at": "2024-01-01"},
                {"name": "mistral:latest", "size": 7_000_000_000, "modified_at": "2024-01-02"},
            ]
        }
        with self._mock_get(tags_data):
            models = self.client.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "llama3:latest")
        self.assertAlmostEqual(models[0].size_gb, 4.0, places=0)

    def test_list_models_server_down(self):
        with self._mock_get(None):
            models = self.client.list_models()
        self.assertEqual(models, [])

    def test_generate_success(self):
        response_data = {"response": "Paris", "done": True}
        with self._mock_post(response_data):
            result = self.client.generate("llama3", "Capital of France?")
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result.response, "Paris")
        self.assertTrue(result.done)
        self.assertEqual(result.error, "")

    def test_generate_server_down(self):
        with self._mock_post(None):
            result = self.client.generate("llama3", "hello")
        self.assertFalse(result.done)
        self.assertNotEqual(result.error, "")

    def test_generate_result_str_success(self):
        r = GenerateResult(model="llama3", prompt="hi", response="hello", done=True)
        self.assertEqual(str(r), "hello")

    def test_generate_result_str_error(self):
        r = GenerateResult(model="llama3", prompt="hi", response="", done=False, error="offline")
        self.assertIn("ERROR", str(r))

    def test_chat_success(self):
        response_data = {"message": {"content": "Hello!"}, "done": True}
        with self._mock_post(response_data):
            result = self.client.chat("llama3", [{"role": "user", "content": "Hi"}])
        self.assertEqual(result.response, "Hello!")

    def test_pull_success(self):
        with self._mock_post({"status": "success"}):
            self.assertTrue(self.client.pull("llama3"))

    def test_pull_failure(self):
        with self._mock_post(None):
            self.assertFalse(self.client.pull("llama3"))

    def test_start_no_binary(self):
        with patch("ai_helper.ai_integrations.shutil.which", return_value=None):
            result = self.client.start()
        self.assertFalse(result)

    def test_ollama_model_str(self):
        m = OllamaModel(name="llama3:latest", size_gb=4.1)
        self.assertIn("llama3", str(m))
        self.assertIn("4.1", str(m))


# ---------------------------------------------------------------------------
# AIAppRegistry
# ---------------------------------------------------------------------------

class TestAIAppRegistry(unittest.TestCase):
    def test_discover_all_offline(self):
        registry = AIAppRegistry(timeout=0.01)
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            statuses = registry.discover()
        self.assertGreater(len(statuses), 0)
        self.assertTrue(all(not s.running for s in statuses))

    def test_discover_ollama_online(self):
        registry = AIAppRegistry(timeout=0.01)

        def _fake_reachable(url, timeout=1.0):
            return "11434" in url

        with patch("ai_helper.ai_integrations._reachable", side_effect=_fake_reachable), \
             patch("ai_helper.ai_integrations._get", return_value=None):
            statuses = registry.discover()

        ollama = next((s for s in statuses if s.name == "Ollama"), None)
        self.assertIsNotNone(ollama)
        self.assertTrue(ollama.running)

    def test_running_returns_only_online(self):
        registry = AIAppRegistry()
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            online = registry.running()
        self.assertEqual(online, [])

    def test_format_status_contains_names(self):
        registry = AIAppRegistry()
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            text = registry.format_status()
        self.assertIn("Ollama", text)
        self.assertIn("ComfyUI", text)
        self.assertIn("LM Studio", text)

    def test_register_custom_app(self):
        registry = AIAppRegistry()
        registry.register("MyAI", "http://localhost:9999/health")
        names = [name for name, _, _ in registry._apps]
        self.assertIn("MyAI", names)

    def test_ai_app_status_str_running(self):
        s = AIAppStatus(name="Ollama", url="http://localhost:11434", running=True, version="0.1")
        self.assertIn("running", str(s))
        self.assertIn("Ollama", str(s))

    def test_ai_app_status_str_not_running(self):
        s = AIAppStatus(name="ComfyUI", url="http://localhost:8188", running=False)
        self.assertIn("not running", str(s))


if __name__ == "__main__":
    unittest.main()
