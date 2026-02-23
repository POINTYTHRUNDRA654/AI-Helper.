"""Tests for ai_helper.gpu_monitor."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_helper.gpu_monitor import (
    DEFAULT_GPU_THRESHOLDS,
    GpuMonitor,
    GpuProcessInfo,
    GpuSnapshot,
)


def _make_snap(
    index=0, name="RTX 4090",
    vram_used=4096.0, vram_total=24576.0,
    temp=65.0, util=80.0,
    power=300.0, power_limit=450.0,
) -> GpuSnapshot:
    free = vram_total - vram_used
    pct = vram_used / vram_total * 100
    return GpuSnapshot(
        index=index, name=name,
        vram_used_mb=vram_used, vram_total_mb=vram_total,
        vram_free_mb=free, vram_percent=pct,
        temperature_c=temp, utilization_percent=util,
        power_draw_w=power, power_limit_w=power_limit,
    )


class TestGpuSnapshot(unittest.TestCase):
    def test_vram_gb_conversion(self):
        snap = _make_snap(vram_used=2048.0, vram_total=8192.0)
        self.assertAlmostEqual(snap.vram_used_gb, 2.0)
        self.assertAlmostEqual(snap.vram_total_gb, 8.0)

    def test_str_contains_name(self):
        snap = _make_snap(name="RTX 3080")
        self.assertIn("RTX 3080", str(snap))

    def test_str_contains_temp(self):
        snap = _make_snap(temp=72.0)
        self.assertIn("72", str(snap))


class TestGpuMonitorAlerts(unittest.TestCase):
    def setUp(self):
        self.monitor = GpuMonitor(thresholds={
            "vram_percent": 80.0,
            "temperature_c": 70.0,
            "utilization_percent": 90.0,
        })

    def test_no_alerts_when_below_threshold(self):
        snap = _make_snap(vram_used=1000.0, vram_total=8192.0, temp=60.0, util=50.0)
        alerts = self.monitor.alerts([snap])
        self.assertEqual(alerts, [])

    def test_vram_alert(self):
        snap = _make_snap(vram_used=7500.0, vram_total=8192.0)
        alerts = self.monitor.alerts([snap])
        self.assertTrue(any("VRAM" in a for a in alerts))

    def test_temperature_alert(self):
        snap = _make_snap(temp=85.0)
        alerts = self.monitor.alerts([snap])
        self.assertTrue(any("temperature" in a.lower() for a in alerts))

    def test_utilisation_alert(self):
        snap = _make_snap(util=95.0)
        alerts = self.monitor.alerts([snap])
        self.assertTrue(any("utilisation" in a.lower() for a in alerts))

    def test_multiple_gpus(self):
        snaps = [
            _make_snap(index=0, temp=60.0),
            _make_snap(index=1, temp=90.0),
        ]
        alerts = self.monitor.alerts(snaps)
        self.assertEqual(len(alerts), 1)
        self.assertIn("GPU 1", alerts[0])


class TestGpuMonitorFormatSnapshots(unittest.TestCase):
    def test_no_gpus(self):
        monitor = GpuMonitor()
        text = monitor.format_snapshots([])
        self.assertIn("No NVIDIA GPUs", text)

    def test_with_gpu(self):
        monitor = GpuMonitor()
        snap = _make_snap(name="Tesla T4")
        text = monitor.format_snapshots([snap])
        self.assertIn("Tesla T4", text)

    def test_with_processes(self):
        monitor = GpuMonitor()
        snap = _make_snap()
        snap.processes = [GpuProcessInfo(pid=1234, name="python", vram_mb=512.0)]
        text = monitor.format_snapshots([snap])
        self.assertIn("python", text)
        self.assertIn("1234", text)


class TestGpuMonitorSmiParsing(unittest.TestCase):
    """Test the nvidia-smi CSV parser without requiring real hardware."""

    def test_parses_valid_smi_output(self):
        smi_output = (
            "0, NVIDIA GeForce RTX 4090, 4096, 24576, 20480, 65, 80, 300.00, 450.00\n"
        )
        monitor = GpuMonitor()
        monitor._nvml = None  # force CLI backend
        with patch("ai_helper.gpu_monitor._smi_available", return_value=True), \
             patch("ai_helper.gpu_monitor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=smi_output)
            snaps = monitor._snapshots_smi()

        self.assertEqual(len(snaps), 1)
        snap = snaps[0]
        self.assertEqual(snap.index, 0)
        self.assertIn("RTX 4090", snap.name)
        self.assertAlmostEqual(snap.vram_used_mb, 4096.0)
        self.assertAlmostEqual(snap.temperature_c, 65.0)
        self.assertAlmostEqual(snap.utilization_percent, 80.0)

    def test_handles_smi_failure(self):
        monitor = GpuMonitor()
        monitor._nvml = None
        with patch("ai_helper.gpu_monitor._smi_available", return_value=True), \
             patch("ai_helper.gpu_monitor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            snaps = monitor._snapshots_smi()
        self.assertEqual(snaps, [])

    def test_no_gpu_returns_empty(self):
        monitor = GpuMonitor()
        monitor._nvml = None
        with patch("ai_helper.gpu_monitor._smi_available", return_value=False):
            snaps = monitor.snapshots()
        self.assertEqual(snaps, [])

    def test_available_false_without_drivers(self):
        monitor = GpuMonitor()
        monitor._nvml = None
        with patch("ai_helper.gpu_monitor._smi_available", return_value=False):
            self.assertFalse(monitor.available)

    def test_available_true_with_smi(self):
        monitor = GpuMonitor()
        monitor._nvml = None
        with patch("ai_helper.gpu_monitor._smi_available", return_value=True):
            self.assertTrue(monitor.available)


if __name__ == "__main__":
    unittest.main()
