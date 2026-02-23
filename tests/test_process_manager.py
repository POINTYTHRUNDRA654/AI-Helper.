"""Tests for ai_helper.process_manager."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_helper.process_manager import ProcessInfo, ProcessManager


def _make_proc(pid=1, name="test", status="sleeping", cpu=0.0, mem=10.0, threads=1):
    return ProcessInfo(
        pid=pid, name=name, status=status,
        cpu_percent=cpu, memory_mb=mem, num_threads=threads,
    )


class TestProcessManager(unittest.TestCase):
    def setUp(self):
        self.pm = ProcessManager(cpu_threshold=50.0, memory_threshold_mb=200.0)

    def test_list_processes_returns_list(self):
        procs = self.pm.list_processes()
        self.assertIsInstance(procs, list)
        self.assertGreater(len(procs), 0)

    def test_high_cpu_processes(self):
        procs = [
            _make_proc(pid=1, cpu=10.0),
            _make_proc(pid=2, cpu=80.0),
        ]
        high = self.pm.high_cpu_processes(procs)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0].pid, 2)

    def test_high_memory_processes(self):
        procs = [
            _make_proc(pid=1, mem=50.0),
            _make_proc(pid=2, mem=500.0),
        ]
        high = self.pm.high_memory_processes(procs)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0].pid, 2)

    def test_find_by_name(self):
        procs = self.pm.find_by_name("python")
        self.assertIsInstance(procs, list)

    def test_summary_contains_total(self):
        procs = [_make_proc(pid=i) for i in range(5)]
        summary = self.pm.summary(procs)
        self.assertIn("5", summary)

    def test_terminate_nonexistent_returns_false(self):
        result = self.pm.terminate(pid=99999999)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
