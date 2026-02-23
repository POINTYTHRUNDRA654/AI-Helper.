"""Tests for new ai_integrations: LMStudioClient, ComfyUIClient, SDWebUIClient."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ai_helper.ai_integrations import (
    ComfyUIClient,
    LMStudioClient,
    SDImage,
    SDWebUIClient,
)


# ---------------------------------------------------------------------------
# LMStudioClient
# ---------------------------------------------------------------------------


class TestLMStudioClient(unittest.TestCase):
    def test_is_running_true(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=True):
            client = LMStudioClient()
            self.assertTrue(client.is_running())

    def test_is_running_false(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            client = LMStudioClient()
            self.assertFalse(client.is_running())

    def test_status_when_running(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=True):
            status = LMStudioClient().status()
        self.assertTrue(status.running)
        self.assertEqual(status.name, "LM Studio")

    def test_list_models(self):
        data = {"data": [{"id": "mistral-7b"}, {"id": "llama3"}]}
        with patch("ai_helper.ai_integrations._get", return_value=data):
            models = LMStudioClient().list_models()
        self.assertEqual(models, ["mistral-7b", "llama3"])

    def test_list_models_empty_when_offline(self):
        with patch("ai_helper.ai_integrations._get", return_value=None):
            models = LMStudioClient().list_models()
        self.assertEqual(models, [])

    def test_chat_success(self):
        response_data = {
            "choices": [{"message": {"content": "Paris"}}]
        }
        with patch("ai_helper.ai_integrations._post", return_value=response_data):
            result = LMStudioClient().chat("What is the capital of France?")
        self.assertEqual(result.response, "Paris")
        self.assertTrue(result.done)

    def test_chat_no_response(self):
        with patch("ai_helper.ai_integrations._post", return_value=None):
            result = LMStudioClient().chat("hello")
        self.assertFalse(result.done)
        self.assertNotEqual(result.error, "")


# ---------------------------------------------------------------------------
# ComfyUIClient
# ---------------------------------------------------------------------------


class TestComfyUIClient(unittest.TestCase):
    def test_is_running_true(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=True):
            self.assertTrue(ComfyUIClient().is_running())

    def test_is_running_false(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            self.assertFalse(ComfyUIClient().is_running())

    def test_status_when_running(self):
        stats = {"system": {"comfyui_version": "1.2.3"}}
        with patch("ai_helper.ai_integrations._get", return_value=stats):
            status = ComfyUIClient().status()
        self.assertTrue(status.running)
        self.assertEqual(status.version, "1.2.3")

    def test_queue_prompt_returns_id(self):
        with patch("ai_helper.ai_integrations._post",
                   return_value={"prompt_id": "abc-123"}):
            pid = ComfyUIClient().queue_prompt({"nodes": {}})
        self.assertEqual(pid, "abc-123")

    def test_queue_prompt_offline(self):
        with patch("ai_helper.ai_integrations._post", return_value=None):
            pid = ComfyUIClient().queue_prompt({})
        self.assertIsNone(pid)

    def test_get_queue(self):
        q = {"queue_running": [], "queue_pending": []}
        with patch("ai_helper.ai_integrations._get", return_value=q):
            result = ComfyUIClient().get_queue()
        self.assertIn("queue_running", result)

    def test_get_history_found(self):
        history = {"abc-123": {"outputs": {}, "status": {"completed": True}}}
        with patch("ai_helper.ai_integrations._get", return_value=history):
            entry = ComfyUIClient().get_history("abc-123")
        self.assertIn("outputs", entry)

    def test_get_history_not_found(self):
        with patch("ai_helper.ai_integrations._get", return_value={}):
            entry = ComfyUIClient().get_history("missing")
        self.assertIsNone(entry)

    def test_interrupt(self):
        with patch("ai_helper.ai_integrations._post", return_value={}):
            ok = ComfyUIClient().interrupt()
        self.assertTrue(ok)

    def test_get_system_stats(self):
        with patch("ai_helper.ai_integrations._get",
                   return_value={"system": {"gpu_info": []}}):
            stats = ComfyUIClient().get_system_stats()
        self.assertIn("system", stats)


# ---------------------------------------------------------------------------
# SDWebUIClient
# ---------------------------------------------------------------------------


class TestSDWebUIClient(unittest.TestCase):
    def _b64_png(self):
        """Minimal valid base64 string to represent a fake PNG."""
        import base64  # noqa: PLC0415
        return base64.b64encode(b"PNG_FAKE_DATA").decode()

    def test_is_running_true(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=True):
            self.assertTrue(SDWebUIClient().is_running())

    def test_is_running_false(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            self.assertFalse(SDWebUIClient().is_running())

    def test_status_when_running(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=True), \
             patch("ai_helper.ai_integrations._get", return_value=None):
            status = SDWebUIClient().status()
        self.assertTrue(status.running)
        self.assertEqual(status.name, "Stable Diffusion WebUI")

    def test_txt2img_success(self):
        img_b64 = self._b64_png()
        resp = {
            "images": [img_b64],
            "info": json.dumps({"all_seeds": [42]}),
        }
        with patch("ai_helper.ai_integrations._post", return_value=resp):
            images = SDWebUIClient().txt2img("a sunset")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].seed, 42)
        self.assertEqual(images[0].prompt, "a sunset")

    def test_txt2img_offline(self):
        with patch("ai_helper.ai_integrations._post", return_value=None):
            images = SDWebUIClient().txt2img("a sunset")
        self.assertEqual(images, [])

    def test_img2img_success(self):
        img_b64 = self._b64_png()
        resp = {"images": [img_b64], "info": "{}"}
        with patch("ai_helper.ai_integrations._post", return_value=resp):
            images = SDWebUIClient().img2img(img_b64, "enhance this")
        self.assertEqual(len(images), 1)

    def test_list_models(self):
        data = [{"title": "v1-5-pruned"}, {"title": "sd-xl-base"}]
        with patch("ai_helper.ai_integrations._get", return_value=data):
            models = SDWebUIClient().list_models()
        self.assertEqual(models, ["v1-5-pruned", "sd-xl-base"])

    def test_list_samplers(self):
        data = [{"name": "Euler a"}, {"name": "DPM++ 2M"}]
        with patch("ai_helper.ai_integrations._get", return_value=data):
            samplers = SDWebUIClient().list_samplers()
        self.assertEqual(samplers, ["Euler a", "DPM++ 2M"])


class TestSDImage(unittest.TestCase):
    def test_save_to_disk(self):
        import base64  # noqa: PLC0415
        from tempfile import NamedTemporaryFile  # noqa: PLC0415
        import os  # noqa: PLC0415
        data = base64.b64encode(b"fake_png_bytes").decode()
        img = SDImage(base64_data=data)
        with NamedTemporaryFile(delete=False, suffix=".png") as f:
            path = f.name
        try:
            ok = img.save(path)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
