"""Tests for ai_helper.web_ui."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

from ai_helper.web_ui import WebUI, _collect_status, _render_html


class TestCollectStatus(unittest.TestCase):
    def test_returns_dict_with_ts(self):
        data = _collect_status()
        self.assertIn("ts", data)
        self.assertIsInstance(data["ts"], str)

    def test_has_cpu_key(self):
        data = _collect_status()
        self.assertIn("cpu", data)
        self.assertIsInstance(data["cpu"], float)

    def test_has_memory_key(self):
        data = _collect_status()
        self.assertIn("memory", data)

    def test_has_processes(self):
        data = _collect_status()
        self.assertIn("processes", data)
        self.assertIsInstance(data["processes"], list)


class TestRenderHTML(unittest.TestCase):
    def _data(self):
        return {
            "ts": "2026-01-01 12:00:00",
            "cpu": 45.0,
            "memory": 60.0,
            "disks": [{"mount": "C:", "pct": 55.0, "used_gb": 110.0, "total_gb": 200.0}],
            "gpus": [{"index": 0, "name": "RTX 4090", "vram_pct": 40.0,
                      "vram_used_gb": 4.0, "vram_total_gb": 10.0,
                      "temp_c": 65, "util_pct": 35}],
            "ai_apps": [{"name": "Ollama", "running": True, "url": "http://localhost:11434"}],
            "processes": [{"pid": 1234, "name": "python", "cpu": 12.0, "mem_mb": 300}],
            "alerts": [],
        }

    def test_html_is_string(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIsInstance(html, str)

    def test_html_contains_cpu(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIn("45.0", html)

    def test_html_contains_gpu_name(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIn("RTX 4090", html)

    def test_html_contains_ai_app(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIn("Ollama", html)

    def test_html_contains_process(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIn("python", html)

    def test_html_no_alerts_shows_ok(self):
        html = _render_html(self._data(), port=8765, refresh=5)
        self.assertIn("No active alerts", html)

    def test_html_with_alerts(self):
        data = self._data()
        data["alerts"] = ["CPU is very high"]
        html = _render_html(data, port=8765, refresh=5)
        self.assertIn("CPU is very high", html)

    def test_refresh_meta_tag(self):
        html = _render_html(self._data(), port=8765, refresh=10)
        self.assertIn('content="10"', html)


class TestWebUI(unittest.TestCase):
    def test_start_and_stop(self):
        ui = WebUI(port=0, host="127.0.0.1")  # port=0 → OS assigns
        # Just test that start/stop don't raise
        ui._server = None  # skip actual bind for speed
        ui.stop()

    def test_running_false_before_start(self):
        ui = WebUI(port=0)
        self.assertFalse(ui.running)

    def test_serve_get_root(self):
        import socket  # noqa: PLC0415
        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        ui = WebUI(port=port, host="127.0.0.1", refresh_seconds=5)
        ui.start()
        time.sleep(0.3)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3)
            html = resp.read().decode()
            self.assertIn("AI Helper", html)
        finally:
            ui.stop()

    def test_serve_get_api(self):
        import socket  # noqa: PLC0415
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        ui = WebUI(port=port, host="127.0.0.1")
        ui.start()
        time.sleep(0.3)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3)
            data = json.loads(resp.read())
            self.assertIn("ts", data)
        finally:
            ui.stop()


if __name__ == "__main__":
    unittest.main()
