"""Tests for ai_helper.communicator — voice integration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_helper.communicator import Communicator, Message, MessageBus, Notifier


class TestMessageBus(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus()

    def test_subscribe_and_publish(self):
        received = []
        self.bus.subscribe("test", received.append)
        self.bus.publish(Message(topic="test", payload="hello"))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload, "hello")

    def test_wildcard_subscriber(self):
        received = []
        self.bus.subscribe("*", received.append)
        self.bus.publish(Message(topic="anything", payload=42))
        self.assertEqual(len(received), 1)

    def test_unsubscribe(self):
        handler = MagicMock()
        self.bus.subscribe("x", handler)
        self.bus.unsubscribe("x", handler)
        self.bus.publish(Message(topic="x", payload="no"))
        handler.assert_not_called()

    def test_history(self):
        for i in range(5):
            self.bus.publish(Message(topic="ev", payload=i))
        self.assertEqual(len(self.bus.history()), 5)
        self.assertEqual(len(self.bus.history(topic="ev")), 5)
        self.assertEqual(len(self.bus.history(limit=3)), 3)

    def test_handler_exception_does_not_crash_bus(self):
        def bad(_msg):
            raise RuntimeError("oops")

        self.bus.subscribe("boom", bad)
        # Must not raise
        self.bus.publish(Message(topic="boom", payload="x"))


class TestCommunicatorVoice(unittest.TestCase):
    """Verify that Communicator.alert() calls speaker.speak() when voice is on."""

    def test_alert_speaks_when_enabled(self):
        mock_speaker = MagicMock()
        comm = Communicator(speaker=mock_speaker, speak_alerts=True)
        comm.alert("High CPU!", source="monitor")
        mock_speaker.speak.assert_called_once()
        spoken = mock_speaker.speak.call_args[0][0]
        self.assertIn("High CPU!", spoken)

    def test_alert_does_not_speak_when_disabled(self):
        mock_speaker = MagicMock()
        comm = Communicator(speaker=mock_speaker, speak_alerts=False)
        comm.alert("High CPU!", source="monitor")
        mock_speaker.speak.assert_not_called()

    def test_no_speaker_no_crash(self):
        comm = Communicator(speak_alerts=False)
        # Must not raise even with no speaker
        comm.alert("test alert")

    def test_speak_alerts_auto_creates_speaker(self):
        """When speak_alerts=True and no speaker is passed, one is created."""
        with patch("ai_helper.voice.Speaker") as MockSpeaker:
            MockSpeaker.return_value = MagicMock()
            comm = Communicator(speak_alerts=True)
            MockSpeaker.assert_called_once()
            self.assertIsNotNone(comm.speaker)

    def test_alert_includes_source_in_spoken_text(self):
        mock_speaker = MagicMock()
        comm = Communicator(speaker=mock_speaker, speak_alerts=True)
        comm.alert("disk full", source="monitor")
        spoken = mock_speaker.speak.call_args[0][0]
        self.assertIn("monitor", spoken)
        self.assertIn("disk full", spoken)


class TestCommunicatorBus(unittest.TestCase):
    def test_publish_and_subscribe(self):
        received = []
        comm = Communicator()
        comm.subscribe("data", received.append)
        comm.publish("data", {"key": "val"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload, {"key": "val"})


if __name__ == "__main__":
    unittest.main()
