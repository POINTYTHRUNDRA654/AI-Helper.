"""Tests for ai_helper.scheduler."""

from __future__ import annotations

import time
import unittest

from ai_helper.scheduler import Task, TaskScheduler


class TestTask(unittest.TestCase):
    def test_run_increments_count(self):
        calls = []
        task = Task(name="t", func=lambda: calls.append(1), interval=60.0)
        task.run()
        self.assertEqual(task.run_count, 1)
        self.assertEqual(len(calls), 1)

    def test_error_increments_error_count(self):
        def bad():
            raise ValueError("boom")

        task = Task(name="bad", func=bad, interval=60.0)
        task.run()
        self.assertEqual(task.error_count, 1)
        self.assertIsNotNone(task.last_error)

    def test_is_due_after_interval(self):
        task = Task(name="t", func=lambda: None, interval=0.01)
        time.sleep(0.05)
        self.assertTrue(task.is_due())

    def test_not_due_immediately(self):
        task = Task(name="t", func=lambda: None, interval=999.0)
        self.assertFalse(task.is_due())

    def test_disabled_task_not_due(self):
        task = Task(name="t", func=lambda: None, interval=0.0, enabled=False)
        self.assertFalse(task.is_due())


class TestTaskScheduler(unittest.TestCase):
    def setUp(self):
        self.sched = TaskScheduler(resolution=0.05)

    def tearDown(self):
        self.sched.stop()

    def test_add_and_remove(self):
        self.sched.add("job1", lambda: None, interval=60.0)
        self.assertIsNotNone(self.sched.get("job1"))
        removed = self.sched.remove("job1")
        self.assertTrue(removed)
        self.assertIsNone(self.sched.get("job1"))

    def test_duplicate_name_raises(self):
        self.sched.add("dup", lambda: None, interval=60.0)
        with self.assertRaises(ValueError):
            self.sched.add("dup", lambda: None, interval=60.0)

    def test_replace_flag(self):
        self.sched.add("rep", lambda: None, interval=60.0)
        self.sched.add("rep", lambda: None, interval=30.0, replace=True)
        self.assertEqual(self.sched.get("rep").interval, 30.0)

    def test_enable_disable(self):
        self.sched.add("tog", lambda: None, interval=60.0, enabled=True)
        self.sched.disable("tog")
        self.assertFalse(self.sched.get("tog").enabled)
        self.sched.enable("tog")
        self.assertTrue(self.sched.get("tog").enabled)

    def test_run_due_executes_due_tasks(self):
        calls = []
        self.sched.add("fast", lambda: calls.append(1), interval=0.01)
        time.sleep(0.05)
        ran = self.sched.run_due()
        self.assertGreaterEqual(ran, 1)
        self.assertGreaterEqual(len(calls), 1)

    def test_run_due_skips_non_due(self):
        calls = []
        self.sched.add("slow", lambda: calls.append(1), interval=9999.0)
        ran = self.sched.run_due()
        self.assertEqual(ran, 0)
        self.assertEqual(len(calls), 0)

    def test_background_thread_runs_tasks(self):
        calls = []
        self.sched.add("bg", lambda: calls.append(1), interval=0.05)
        self.sched.start()
        time.sleep(0.4)
        self.sched.stop()
        self.assertGreaterEqual(len(calls), 1)

    def test_status_returns_list(self):
        self.sched.add("s1", lambda: None, interval=10.0)
        self.sched.add("s2", lambda: None, interval=20.0)
        status = self.sched.status()
        self.assertEqual(len(status), 2)
        names = [s.name for s in status]
        self.assertIn("s1", names)
        self.assertIn("s2", names)

    def test_start_stop_idempotent(self):
        self.sched.start()
        self.assertTrue(self.sched.running)
        self.sched.start()  # second start is a no-op
        self.sched.stop()
        self.assertFalse(self.sched.running)


if __name__ == "__main__":
    unittest.main()
