"""Tests for ai_helper.notification_center."""

from __future__ import annotations

import time
import unittest

from ai_helper.notification_center import NotificationCenter, NotificationRecord


class TestNotificationCenter(unittest.TestCase):
    def _nc(self, **kwargs) -> NotificationCenter:
        return NotificationCenter(**kwargs)

    def test_basic_notify_fires(self):
        nc = self._nc()
        rec = nc.notify("CPU high", source="monitor", urgency="critical")
        self.assertFalse(rec.suppressed)
        self.assertEqual(rec.source, "monitor")
        self.assertEqual(rec.urgency, "critical")

    def test_deduplication_suppresses_repeat(self):
        nc = self._nc(dedup_seconds=60)
        nc.notify("CPU high", source="monitor")
        rec2 = nc.notify("CPU high", source="monitor")
        self.assertTrue(rec2.suppressed)

    def test_dedup_after_cooldown_fires_again(self):
        nc = self._nc(dedup_seconds=0.05, throttle_seconds=0)
        nc.notify("CPU high", source="monitor")
        time.sleep(0.1)
        rec2 = nc.notify("CPU high", source="monitor")
        self.assertFalse(rec2.suppressed)

    def test_throttle_suppresses_different_message_same_topic(self):
        nc = self._nc(throttle_seconds=60)
        nc.notify("msg1", source="monitor", topic="myapp")
        rec2 = nc.notify("msg2", source="monitor", topic="myapp")
        self.assertTrue(rec2.suppressed)

    def test_throttle_different_topic_fires(self):
        nc = self._nc(throttle_seconds=60)
        nc.notify("msg1", source="monitor", topic="topic_a")
        rec2 = nc.notify("msg2", source="monitor", topic="topic_b")
        self.assertFalse(rec2.suppressed)

    def test_escalation_raises_urgency(self):
        nc = self._nc(
            dedup_seconds=0,
            throttle_seconds=0,
            escalate_count=3,
            escalate_window=60,
        )
        for _ in range(3):
            time.sleep(0.01)
            rec = nc.notify("CPU high", source="monitor", urgency="normal")
        self.assertEqual(rec.urgency, "critical")
        self.assertTrue(rec.escalated)

    def test_no_escalation_below_count(self):
        nc = self._nc(
            dedup_seconds=0,
            throttle_seconds=0,
            escalate_count=5,
            escalate_window=60,
        )
        for _ in range(2):
            time.sleep(0.01)
            rec = nc.notify("CPU high", source="monitor", urgency="normal")
        self.assertFalse(rec.escalated)

    def test_history_records_all(self):
        nc = self._nc()
        nc.notify("a", source="s1")
        nc.notify("b", source="s2")
        self.assertEqual(len(nc.history), 2)

    def test_active_alerts_excludes_suppressed(self):
        nc = self._nc(dedup_seconds=60)
        nc.notify("msg", source="s")
        nc.notify("msg", source="s")  # suppressed
        active = nc.active_alerts()
        self.assertEqual(len(active), 1)

    def test_format_history_contains_message(self):
        nc = self._nc()
        nc.notify("High CPU", source="monitor", urgency="critical")
        text = nc.format_history()
        self.assertIn("High CPU", text)

    def test_format_history_urgency_filter(self):
        nc = self._nc(dedup_seconds=0, throttle_seconds=0)
        nc.notify("critical msg", source="s", urgency="critical")
        time.sleep(0.01)
        nc.notify("normal msg", source="s", urgency="normal")
        text = nc.format_history(urgency="critical")
        self.assertIn("critical msg", text)
        self.assertNotIn("normal msg", text)

    def test_format_history_source_filter(self):
        nc = self._nc(dedup_seconds=0, throttle_seconds=0)
        nc.notify("from monitor", source="monitor")
        time.sleep(0.01)
        nc.notify("from gpu", source="gpu_monitor")
        text = nc.format_history(source="monitor")
        self.assertIn("from monitor", text)
        self.assertNotIn("from gpu", text)

    def test_stats_string(self):
        nc = self._nc()
        nc.notify("msg", source="s", urgency="critical")
        stats = nc.stats()
        self.assertIn("active", stats)
        self.assertIn("critical", stats)

    def test_on_notify_callback_called(self):
        received = []
        nc = self._nc(on_notify=received.append)
        nc.notify("test", source="s")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].message, "test")

    def test_on_notify_not_called_for_suppressed(self):
        received = []
        nc = self._nc(dedup_seconds=60, on_notify=received.append)
        nc.notify("dup", source="s")
        nc.notify("dup", source="s")
        self.assertEqual(len(received), 1)

    def test_clear_history(self):
        nc = self._nc()
        nc.notify("a", source="s")
        nc.clear_history()
        self.assertEqual(len(nc.history), 0)
        # Should fire again after clear
        rec = nc.notify("a", source="s")
        self.assertFalse(rec.suppressed)

    def test_notification_record_str(self):
        rec = NotificationRecord(message="test", source="monitor", urgency="critical",
                                 topic="alert")
        text = str(rec)
        self.assertIn("test", text)
        self.assertIn("monitor", text)

    def test_notification_record_str_escalated(self):
        rec = NotificationRecord(message="x", source="s", urgency="critical",
                                 topic="t", escalated=True)
        self.assertIn("ESCALATED", str(rec))

    def test_history_maxsize(self):
        nc = self._nc(history_size=5, dedup_seconds=0, throttle_seconds=0)
        for i in range(10):
            time.sleep(0.01)
            nc.notify(f"msg {i}", source="s", topic=f"t{i}")
        self.assertLessEqual(len(nc.history), 5)


if __name__ == "__main__":
    unittest.main()
