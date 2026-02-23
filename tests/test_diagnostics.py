"""Tests for ai_helper.diagnostics."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from ai_helper.diagnostics import (
    CheckResult,
    DiagnosticsReport,
    _check_callable,
    _check_import,
    _check_import_metadata,
    _check_optional_import,
    _check_core_modules,
    _check_core_classes,
    _check_version_declared,
    _check_python_version,
    _version_gte,
    run_diagnostics,
)


class TestVersionGte(unittest.TestCase):
    def test_equal(self):
        self.assertTrue(_version_gte("5.9.0", "5.9.0"))

    def test_greater(self):
        self.assertTrue(_version_gte("5.10.0", "5.9.0"))

    def test_less(self):
        self.assertFalse(_version_gte("5.8.0", "5.9.0"))

    def test_major_wins(self):
        self.assertTrue(_version_gte("6.0.0", "5.99.99"))

    def test_unparseable_returns_true(self):
        # Both sides invalid — _parse returns (0,) for each, so they're equal → True
        self.assertTrue(_version_gte("bad", "bad"))


class TestCheckResult(unittest.TestCase):
    def test_str_pass(self):
        r = CheckResult(name="test", passed=True, message="OK")
        self.assertIn("[PASS]", str(r))
        self.assertIn("OK", str(r))

    def test_str_fail(self):
        r = CheckResult(name="test", passed=False, message="broken", detail="traceback")
        self.assertIn("[FAIL]", str(r))
        self.assertIn("broken", str(r))
        self.assertIn("traceback", str(r))


class TestDiagnosticsReport(unittest.TestCase):
    def test_all_pass(self):
        r = DiagnosticsReport(
            checks=[
                CheckResult("a", True, "ok"),
                CheckResult("b", True, "ok"),
            ]
        )
        self.assertTrue(r.passed)
        self.assertEqual(r.fail_count, 0)

    def test_some_fail(self):
        r = DiagnosticsReport(
            checks=[
                CheckResult("a", True, "ok"),
                CheckResult("b", False, "broken"),
            ]
        )
        self.assertFalse(r.passed)
        self.assertEqual(r.fail_count, 1)

    def test_str_contains_result(self):
        r = DiagnosticsReport(
            checks=[CheckResult("x", True, "good")]
        )
        text = str(r)
        self.assertIn("PASSED", text)


class TestCheckImportMetadata(unittest.TestCase):
    def test_installed_package(self):
        result = _check_import_metadata("psutil", min_version="5.9.0")
        self.assertTrue(result.passed, result.message)

    def test_not_installed(self):
        result = _check_import_metadata("nonexistent_xyz_pkg_123")
        self.assertFalse(result.passed)

    def test_version_below_min(self):
        with patch("importlib.metadata.version", return_value="0.0.1"):
            result = _check_import_metadata("psutil", min_version="5.9.0")
        self.assertFalse(result.passed)


class TestCheckImport(unittest.TestCase):
    def test_existing_module(self):
        result = _check_import("psutil", min_version="5.9.0")
        self.assertTrue(result.passed, result.message)

    def test_missing_module(self):
        result = _check_import("nonexistent_pkg_xyz_123")
        self.assertFalse(result.passed)

    def test_version_pass(self):
        result = _check_import("psutil", min_version="1.0.0")
        self.assertTrue(result.passed)

    def test_version_fail(self):
        # Patch metadata to return an old version
        with patch("importlib.metadata.version", return_value="0.0.1"):
            result = _check_import("psutil", min_version="5.9.0")
        self.assertFalse(result.passed)

class TestCheckOptionalImport(unittest.TestCase):
    def test_present(self):
        # psutil is installed, treat as optional
        result = _check_optional_import("psutil")
        self.assertTrue(result.passed)

    def test_absent_still_passes(self):
        result = _check_optional_import("nonexistent_optional_pkg_xyz")
        self.assertTrue(result.passed)
        self.assertIn("not installed", result.message)


class TestCheckCallable(unittest.TestCase):
    def test_no_exception(self):
        result = _check_callable("ok_check", lambda: None)
        self.assertTrue(result.passed)

    def test_exception_fails(self):
        def boom():
            raise ValueError("error!")

        result = _check_callable("bad_check", boom)
        self.assertFalse(result.passed)
        self.assertIn("ValueError", result.detail)


class TestCheckCoreModules(unittest.TestCase):
    def test_all_core_modules_importable(self):
        results = _check_core_modules()
        self.assertGreater(len(results), 10)
        failures = [r for r in results if not r.passed]
        self.assertEqual(failures, [], f"Module import failures: {failures}")


class TestCheckCoreClasses(unittest.TestCase):
    def test_core_classes_pass(self):
        results = _check_core_classes()
        self.assertGreater(len(results), 5)
        failures = [r for r in results if not r.passed]
        self.assertEqual(failures, [], f"Class smoke-test failures: {failures}")


class TestCheckVersionDeclared(unittest.TestCase):
    def test_version_present(self):
        result = _check_version_declared()
        self.assertTrue(result.passed, result.message)
        self.assertIn("0.1.0", result.message)


class TestCheckPythonVersion(unittest.TestCase):
    def test_current_python_passes(self):
        # We require 3.10+; the test environment should satisfy this.
        result = _check_python_version()
        self.assertTrue(result.passed, result.message)

    def test_old_python_fails(self):
        from collections import namedtuple
        FakeVI = namedtuple("version_info", ["major", "minor", "micro"])
        with patch.object(sys, "version_info", FakeVI(3, 9, 0)):
            result = _check_python_version()
        self.assertFalse(result.passed)


class TestRunDiagnostics(unittest.TestCase):
    def test_run_returns_report_and_bool(self):
        report, passed = run_diagnostics(verbose=False)
        self.assertIsInstance(report, DiagnosticsReport)
        self.assertIsInstance(passed, bool)

    def test_run_passes_in_healthy_env(self):
        _report, passed = run_diagnostics(verbose=False)
        self.assertTrue(passed, f"Diagnostics failed:\n{_report}")

    def test_run_verbose_prints(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_diagnostics(verbose=True)
        output = buf.getvalue()
        self.assertIn("PASS", output)


if __name__ == "__main__":
    unittest.main()
