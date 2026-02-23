"""Tests for the updated orchestrator (notification_center + memory integration)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from ai_helper.memory import Memory
from ai_helper.notification_center import NotificationCenter
from ai_helper.orchestrator import Orchestrator


class TestOrchestratorWithMemoryAndNC(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db = Path(self._tmp.name) / "mem.db"
        self.mem = Memory(db_path=self.db)
        self.nc = NotificationCenter(dedup_seconds=0, throttle_seconds=0)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_orchestrator(self):
        monitor = MagicMock()
        monitor.snapshot.return_value = MagicMock(
            cpu_percent=45.0,
            memory_percent=50.0,
            disk_partitions=[],
        )
        monitor.alerts.return_value = []
        monitor.format_snapshot.return_value = "snap"

        proc_mgr = MagicMock()
        proc_mgr.list_processes.return_value = []
        proc_mgr.high_cpu_processes.return_value = []
        proc_mgr.high_memory_processes.return_value = []
        proc_mgr.summary.return_value = "procs"

        gpu = MagicMock()
        gpu.available = False

        ai_reg = MagicMock()
        ai_reg.running.return_value = []

        return Orchestrator(
            monitor=monitor,
            process_manager=proc_mgr,
            gpu_monitor=gpu,
            ai_registry=ai_reg,
            notification_center=self.nc,
            memory=self.mem,
        )

    def test_tick_no_alerts(self):
        orch = self._make_orchestrator()
        orch.tick()
        # No alerts fired
        self.assertEqual(len(self.nc.active_alerts()), 0)

    def test_tick_system_alert_fires_nc(self):
        orch = self._make_orchestrator()
        orch.monitor.alerts.return_value = ["CPU is 97%"]
        orch.tick()
        alerts = self.nc.active_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertIn("CPU", alerts[0].message)

    def test_tick_system_alert_records_in_memory(self):
        orch = self._make_orchestrator()
        orch.monitor.alerts.return_value = ["Memory is 95%"]
        orch.tick()
        self.assertGreater(self.mem.anomaly_count(), 0)

    def test_tick_process_alert_fires_nc(self):
        proc = MagicMock()
        proc.name = "chrome"
        proc.pid = 1234
        proc.cpu_percent = 95.0
        proc.memory_mb = 100.0

        orch = self._make_orchestrator()
        orch.process_manager.high_cpu_processes.return_value = [proc]
        orch.tick()

        alerts = self.nc.active_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertIn("chrome", alerts[0].message)

    def test_alert_deduplication_via_nc(self):
        orch = self._make_orchestrator()
        orch.notification_center = NotificationCenter(dedup_seconds=60,
                                                      throttle_seconds=0)
        orch.monitor.alerts.return_value = ["CPU is high"]
        orch.tick()
        orch.tick()

        active = orch.notification_center.active_alerts()
        self.assertEqual(len(active), 1)

    def test_orchestrator_has_memory(self):
        orch = self._make_orchestrator()
        self.assertIsNotNone(orch.memory)

    def test_orchestrator_has_notification_center(self):
        orch = self._make_orchestrator()
        self.assertIsNotNone(orch.notification_center)

    def test_start_and_stop(self):
        orch = self._make_orchestrator()
        orch.start()
        self.assertTrue(orch.running)
        orch.stop()
        self.assertFalse(orch.running)


if __name__ == "__main__":
    unittest.main()
