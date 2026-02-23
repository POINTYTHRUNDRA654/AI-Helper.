"""Tests for ai_helper.clipboard_monitor."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from ai_helper.clipboard_monitor import (
    ClipboardEvent,
    ClipboardMonitor,
    classify,
    read_clipboard,
)


class TestClassify(unittest.TestCase):
    def test_windows_path(self):
        self.assertEqual(classify(r"C:\Users\Alice\Documents\report.docx"), "file_path")

    def test_unix_path(self):
        self.assertEqual(classify("/home/alice/notes.txt"), "file_path")

    def test_url(self):
        self.assertEqual(classify("https://example.com/page"), "url")

    def test_python_traceback(self):
        tb = "Traceback (most recent call last):\n  File 'x.py'\nValueError: bad"
        self.assertEqual(classify(tb), "error")

    def test_error_keyword(self):
        self.assertEqual(classify("TypeError: expected str got int"), "error")

    def test_pip_command(self):
        self.assertEqual(classify("pip install requests"), "command")

    def test_python_command(self):
        self.assertEqual(classify("python -m ai_helper --daemon"), "command")

    def test_git_command(self):
        self.assertEqual(classify("git commit -m 'fix bug'"), "command")

    def test_plain_text(self):
        self.assertEqual(classify("Hello, how are you today?"), "text")

    def test_short_text(self):
        self.assertEqual(classify("hi"), "text")


class TestClipboardEvent(unittest.TestCase):
    def test_str_contains_kind(self):
        event = ClipboardEvent(text="python -m test", kind="command")
        self.assertIn("command", str(event))

    def test_str_preview_truncated(self):
        event = ClipboardEvent(text="x" * 200, kind="text")
        self.assertLessEqual(len(str(event)), 200)


class TestClipboardMonitor(unittest.TestCase):
    def test_callback_fired_on_change(self):
        events = []
        mon = ClipboardMonitor(callback=events.append, poll_interval=0.05)

        clipboard_contents = ["first text"]
        with patch("ai_helper.clipboard_monitor.read_clipboard",
                   side_effect=lambda: clipboard_contents[0]):
            mon.start()
            time.sleep(0.2)
            clipboard_contents[0] = "second text"
            time.sleep(0.2)
            mon.stop()

        kinds = [e.kind for e in events]
        self.assertGreaterEqual(len(events), 1)

    def test_no_duplicate_events(self):
        events = []
        mon = ClipboardMonitor(callback=events.append, poll_interval=0.05)

        with patch("ai_helper.clipboard_monitor.read_clipboard", return_value="same text"):
            mon.start()
            time.sleep(0.3)
            mon.stop()

        # Should fire exactly once even though content never changed
        self.assertEqual(len(events), 1)

    def test_start_stop(self):
        mon = ClipboardMonitor(callback=lambda e: None, poll_interval=0.05)
        with patch("ai_helper.clipboard_monitor.read_clipboard", return_value=None):
            mon.start()
            self.assertTrue(mon.running)
            mon.stop()
            self.assertFalse(mon.running)

    def test_short_content_ignored(self):
        events = []
        mon = ClipboardMonitor(callback=events.append, poll_interval=0.05, min_length=10)
        with patch("ai_helper.clipboard_monitor.read_clipboard", return_value="hi"):
            mon.start()
            time.sleep(0.2)
            mon.stop()
        self.assertEqual(events, [])

    def test_read_clipboard_returns_none_on_error(self):
        with patch("ai_helper.clipboard_monitor._read_clipboard_pyperclip", return_value=None), \
             patch("ai_helper.clipboard_monitor._SYSTEM", "UnknownOS"):
            result = read_clipboard()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
