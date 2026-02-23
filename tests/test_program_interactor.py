"""Tests for ai_helper.program_interactor."""

from __future__ import annotations

import signal
import unittest
from unittest.mock import MagicMock, patch

from ai_helper.program_interactor import (
    AppInfo,
    CommunicateResult,
    LaunchResult,
    ProgramInteractor,
)


class TestLaunchResult(unittest.TestCase):
    def test_str_success(self):
        r = LaunchResult(command="echo", pid=123, success=True)
        self.assertIn("123", str(r))

    def test_str_failure(self):
        r = LaunchResult(command="bad", pid=None, success=False, error="not found")
        self.assertIn("not found", str(r))


class TestProgramInteractor(unittest.TestCase):
    def setUp(self):
        self.pi = ProgramInteractor(default_timeout=5.0)

    def test_communicate_echo(self):
        result = self.pi.communicate("echo", args=["hello"])
        self.assertIsInstance(result, CommunicateResult)
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)

    def test_communicate_nonexistent_command(self):
        result = self.pi.communicate("nonexistent_program_xyz_404")
        self.assertIsInstance(result, CommunicateResult)
        self.assertNotEqual(result.returncode, 0)

    def test_communicate_with_input(self):
        result = self.pi.communicate("cat", input_data="hello from stdin\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello from stdin", result.stdout)

    def test_communicate_timeout(self):
        result = self.pi.communicate("sleep", args=["10"], timeout=0.1)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.returncode)

    def test_launch_nonexistent_returns_failure(self):
        result = self.pi.launch("nonexistent_program_xyz_404")
        self.assertFalse(result.success)
        self.assertIsNone(result.pid)

    def test_launch_returns_launch_result(self):
        # Launch a quick command; we just check the type
        result = self.pi.launch("echo", args=["hi"], detach=False)
        self.assertIsInstance(result, LaunchResult)

    def test_find_running_python(self):
        # The test process itself is Python, so there must be at least one.
        found = self.pi.find_running("python")
        self.assertGreater(len(found), 0)

    def test_send_signal_nonexistent_returns_false(self):
        result = self.pi.send_signal(pid=99999999, sig=signal.SIGTERM)
        self.assertFalse(result)

    def test_terminate_nonexistent_returns_false(self):
        result = self.pi.terminate(pid=99999999)
        self.assertFalse(result)

    def test_list_installed_returns_list(self):
        apps = self.pi.list_installed()
        self.assertIsInstance(apps, list)
        # On any platform there should be at least a few executables
        self.assertGreater(len(apps), 0)

    def test_communicate_result_str(self):
        r = CommunicateResult(command="ls", returncode=0, stdout="file.txt\n", stderr="")
        self.assertIn("ls", str(r))

    def test_communicate_result_timeout_str(self):
        r = CommunicateResult(command="sleep", returncode=None, stdout="", stderr="", timed_out=True)
        self.assertIn("TIMEOUT", str(r))


if __name__ == "__main__":
    unittest.main()
