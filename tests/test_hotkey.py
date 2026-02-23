"""Tests for ai_helper.hotkey."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_helper.hotkey import HotkeyManager, HOTKEY_ASK, HOTKEY_STATUS


class TestHotkeyManager(unittest.TestCase):
    def test_bindings_info_contains_combos(self):
        mgr = HotkeyManager()
        info = mgr.bindings_info()
        self.assertIn(HOTKEY_ASK, info)
        self.assertIn(HOTKEY_STATUS, info)

    def test_start_returns_false_without_pynput(self):
        mgr = HotkeyManager()
        with patch.dict("sys.modules", {"pynput": None, "pynput.keyboard": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = mgr.start()
        self.assertFalse(result)

    def test_start_with_pynput(self):
        mock_pynput = MagicMock()
        mock_hotkeys_instance = MagicMock()
        mock_pynput.keyboard.GlobalHotKeys.return_value = mock_hotkeys_instance
        mock_hotkeys_instance.run = lambda: None

        mgr = HotkeyManager()
        with patch.dict("sys.modules", {"pynput": mock_pynput,
                                         "pynput.keyboard": mock_pynput.keyboard}):
            result = mgr.start()

        self.assertTrue(result)

    def test_custom_bindings(self):
        called = []
        mgr = HotkeyManager(hotkeys={"<ctrl>+<alt>+x": lambda: called.append(True)})
        info = mgr.bindings_info()
        self.assertIn("<ctrl>+<alt>+x", info)

    def test_stop_does_not_crash_when_not_started(self):
        mgr = HotkeyManager()
        mgr.stop()   # Should not raise

    def test_running_false_before_start(self):
        mgr = HotkeyManager()
        self.assertFalse(mgr.running)

    def test_safe_call_catches_exception(self):
        def bad():
            raise RuntimeError("boom")
        HotkeyManager._safe_call(bad)   # Should not raise


if __name__ == "__main__":
    unittest.main()
