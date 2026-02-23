"""Tests for ai_helper.monitor."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from ai_helper.monitor import (
    DEFAULT_THRESHOLDS,
    DiskStats,
    NetworkStats,
    SystemMonitor,
    SystemSnapshot,
)


def _make_snapshot(cpu=10.0, mem=20.0, disk_pct=30.0) -> SystemSnapshot:
    return SystemSnapshot(
        timestamp=time.time(),
        cpu_percent=cpu,
        memory_percent=mem,
        memory_used_mb=2048.0,
        memory_total_mb=8192.0,
        disks=[DiskStats(path="/", total_gb=500.0, used_gb=150.0, free_gb=350.0, percent=disk_pct)],
        network=NetworkStats(bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20),
    )


class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = SystemMonitor()

    def test_snapshot_returns_snapshot(self):
        snap = self.monitor.snapshot()
        self.assertIsInstance(snap, SystemSnapshot)
        self.assertGreaterEqual(snap.cpu_percent, 0.0)
        self.assertGreaterEqual(snap.memory_percent, 0.0)

    def test_no_alerts_when_below_threshold(self):
        snap = _make_snapshot(cpu=10.0, mem=20.0, disk_pct=30.0)
        alerts = self.monitor.alerts(snap)
        self.assertEqual(alerts, [])

    def test_cpu_alert(self):
        monitor = SystemMonitor(thresholds={"cpu": 50.0, "memory": 90.0, "disk": 90.0})
        snap = _make_snapshot(cpu=90.0)
        alerts = monitor.alerts(snap)
        self.assertTrue(any("CPU" in a for a in alerts))

    def test_memory_alert(self):
        monitor = SystemMonitor(thresholds={"cpu": 90.0, "memory": 50.0, "disk": 90.0})
        snap = _make_snapshot(mem=75.0)
        alerts = monitor.alerts(snap)
        self.assertTrue(any("memory" in a.lower() for a in alerts))

    def test_disk_alert(self):
        monitor = SystemMonitor(thresholds={"cpu": 90.0, "memory": 90.0, "disk": 50.0})
        snap = _make_snapshot(disk_pct=75.0)
        alerts = monitor.alerts(snap)
        self.assertTrue(any("disk" in a.lower() for a in alerts))

    def test_format_snapshot_contains_cpu(self):
        snap = _make_snapshot()
        text = self.monitor.format_snapshot(snap)
        self.assertIn("CPU", text)
        self.assertIn("Memory", text)

    def test_alerts_takes_fresh_snapshot_when_none(self):
        # Should not raise; takes internal snapshot
        alerts = self.monitor.alerts()
        self.assertIsInstance(alerts, list)


if __name__ == "__main__":
    unittest.main()
