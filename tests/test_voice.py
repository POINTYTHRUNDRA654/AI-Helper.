"""Tests for ai_helper.voice."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from ai_helper.voice import Speaker, VoiceSettings, _cli_fallback_available


class TestVoiceSettings(unittest.TestCase):
    def test_defaults(self):
        s = VoiceSettings()
        self.assertEqual(s.rate, 175)
        self.assertAlmostEqual(s.volume, 1.0)
        self.assertIsNone(s.voice_id)

    def test_custom(self):
        s = VoiceSettings(rate=120, volume=0.5, voice_id="english")
        self.assertEqual(s.rate, 120)
        self.assertAlmostEqual(s.volume, 0.5)
        self.assertEqual(s.voice_id, "english")


class TestSpeakerDisabled(unittest.TestCase):
    """When enabled=False the speaker must be silent and non-blocking."""

    def setUp(self):
        self.speaker = Speaker(enabled=False)

    def tearDown(self):
        self.speaker.shutdown()

    def test_speak_returns_immediately(self):
        # Must not raise and must return quickly
        self.speaker.speak("hello")

    def test_speak_now_returns_false(self):
        result = self.speaker.speak_now("hello", timeout=2.0)
        self.assertFalse(result)

    def test_stop_does_not_raise(self):
        self.speaker.stop()


class TestSpeakerEnabled(unittest.TestCase):
    """Enabled speaker with pyttsx3 mocked out so no audio hardware is needed."""

    def _make_speaker(self):
        """Return a Speaker whose pyttsx3 engine is fully mocked."""
        mock_engine = MagicMock()
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch("ai_helper.voice._try_import_pyttsx3", return_value=mock_pyttsx3):
            speaker = Speaker(settings=VoiceSettings(rate=200, volume=0.8), enabled=True)
        # Inject the mock engine directly so the worker thread can use it.
        speaker._pyttsx3 = mock_pyttsx3
        speaker._engine = mock_engine
        return speaker, mock_engine

    def test_speak_enqueues(self):
        speaker, mock_engine = self._make_speaker()
        try:
            speaker.speak("test message")
            # Give the worker thread a moment to dequeue.
            time.sleep(0.3)
            mock_engine.say.assert_called()
        finally:
            speaker.shutdown()

    def test_speak_now_blocking(self):
        speaker, mock_engine = self._make_speaker()
        try:
            result = speaker.speak_now("blocking test", timeout=5.0)
            # speak_now returns True when the engine completes (or mock doesn't block).
            self.assertIsInstance(result, bool)
        finally:
            speaker.shutdown()

    def test_stop_clears_queue(self):
        speaker, _ = self._make_speaker()
        try:
            for _ in range(10):
                speaker._queue.put("item")
            speaker.stop()
            self.assertEqual(speaker._queue.qsize(), 0)
        finally:
            speaker.shutdown()

    def test_list_voices_no_pyttsx3(self):
        speaker = Speaker(enabled=False)
        speaker._pyttsx3 = None
        voices = speaker.list_voices()
        self.assertIsInstance(voices, list)
        self.assertEqual(voices, [])
        speaker.shutdown()

    def test_list_voices_with_pyttsx3(self):
        mock_voice = MagicMock()
        mock_voice.name = "Test Voice"
        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice]
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        speaker = Speaker(enabled=False)
        speaker._pyttsx3 = mock_pyttsx3
        voices = speaker.list_voices()
        self.assertIn("Test Voice", voices)
        speaker.shutdown()


class TestSpeakerCliFallback(unittest.TestCase):
    """Test CLI fallback path when pyttsx3 is unavailable."""

    def test_utter_cli_linux_espeak(self):
        speaker = Speaker(enabled=True)
        speaker._pyttsx3 = None
        speaker._engine = None
        with patch("ai_helper.voice._SYSTEM", "Linux"), \
             patch("ai_helper.voice.shutil.which", return_value="/usr/bin/espeak"), \
             patch("ai_helper.voice.subprocess.run") as mock_run:
            speaker._utter_cli("hello espeak")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], "espeak")
            self.assertIn("hello espeak", cmd)
        speaker.shutdown()

    def test_utter_cli_macos_say(self):
        speaker = Speaker(enabled=True)
        speaker._pyttsx3 = None
        speaker._engine = None
        with patch("ai_helper.voice._SYSTEM", "Darwin"), \
             patch("ai_helper.voice.shutil.which", return_value="/usr/bin/say"), \
             patch("ai_helper.voice.subprocess.run") as mock_run:
            speaker._utter_cli("hello mac")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], "say")
        speaker.shutdown()

    def test_utter_cli_windows_powershell(self):
        speaker = Speaker(enabled=True)
        speaker._pyttsx3 = None
        speaker._engine = None
        with patch("ai_helper.voice._SYSTEM", "Windows"), \
             patch("ai_helper.voice.subprocess.run") as mock_run:
            speaker._utter_cli("hello windows")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], "powershell")
        speaker.shutdown()

    def test_quotes_sanitised(self):
        """Double-quotes in text must be converted to single-quotes for shell safety."""
        speaker = Speaker(enabled=True)
        speaker._pyttsx3 = None
        speaker._engine = None
        with patch("ai_helper.voice._SYSTEM", "Linux"), \
             patch("ai_helper.voice.shutil.which", return_value="/usr/bin/espeak"), \
             patch("ai_helper.voice.subprocess.run") as mock_run:
            speaker._utter_cli('say "hello"')
            call_args = mock_run.call_args[0][0]
            spoken_text = call_args[-1]
            self.assertNotIn('"', spoken_text)
        speaker.shutdown()


class TestPsRate(unittest.TestCase):
    def test_default_rate_maps_to_zero(self):
        s = Speaker(settings=VoiceSettings(rate=200), enabled=False)
        self.assertEqual(s._ps_rate(), 0)
        s.shutdown()

    def test_slow_rate_is_negative(self):
        s = Speaker(settings=VoiceSettings(rate=100), enabled=False)
        self.assertLess(s._ps_rate(), 0)
        s.shutdown()

    def test_fast_rate_is_positive(self):
        s = Speaker(settings=VoiceSettings(rate=400), enabled=False)
        self.assertGreater(s._ps_rate(), 0)
        s.shutdown()

    def test_rate_clamped(self):
        s = Speaker(settings=VoiceSettings(rate=9999), enabled=False)
        self.assertEqual(s._ps_rate(), 10)
        s.shutdown()


if __name__ == "__main__":
    unittest.main()
