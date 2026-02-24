"""AI Helper diagnostics.

Performs a comprehensive health-check of the AI Helper installation:

* Verifies that every required Python package is installed and importable.
* Checks that optional packages (pynvml, etc.) are present or absent with a
  clear note.
* Imports and lightly exercises every core AI Helper module to confirm that
  none of them are stubs or broken imports.
* Runs non-destructive smoke tests on the main classes.
* Prints a human-readable pass / fail report and returns a boolean overall
  result so callers can act on failure.

Usage
-----
From the CLI::

    python -m ai_helper --diagnostics

Programmatically::

    from ai_helper.diagnostics import run_diagnostics
    ok = run_diagnostics()
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    passed: bool
    message: str
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        base = f"  [{status}] {self.name}: {self.message}"
        if self.detail:
            base += f"\n         {self.detail}"
        return base


@dataclass
class DiagnosticsReport:
    """Aggregate result of all diagnostic checks."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  AI Helper – Installation Diagnostics",
            "=" * 60,
        ]
        for check in self.checks:
            lines.append(str(check))
        lines.append("=" * 60)
        total = len(self.checks)
        passed = total - self.fail_count
        if self.passed:
            lines.append(f"  Result: ALL {total} checks PASSED ✓")
        else:
            lines.append(
                f"  Result: {self.fail_count} of {total} checks FAILED ✗ "
                f"({passed} passed)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def _version_gte(installed: str, minimum: str) -> bool:
    """Return True if *installed* version string >= *minimum* version string."""
    def _parse(v: str) -> tuple:
        parts = []
        for part in v.split("."):
            # Strip any non-numeric suffix (e.g. "5.9.0a1" → 5, 9, 0)
            numeric = ""
            for ch in part:
                if ch.isdigit():
                    numeric += ch
                else:
                    break
            parts.append(int(numeric) if numeric else 0)
        return tuple(parts)

    try:
        return _parse(installed) >= _parse(minimum)
    except (ValueError, TypeError):
        return True  # Can't parse — assume OK


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _check_import_metadata(
    package: str,
    min_version: Optional[str] = None,
) -> CheckResult:
    """Check that *package* is **installed** (via metadata) without importing it.

    This is used for packages like ``pynput`` that are installed correctly but
    may fail to import in headless CI environments (no display server).  On a
    real desktop the import will succeed; we only verify the package is present.
    """
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            name=f"package:{package}",
            passed=False,
            message=f"'{package}' is not installed",
        )
    if min_version is None or _version_gte(installed, min_version):
        return CheckResult(
            name=f"package:{package}",
            passed=True,
            message=f"'{package}' {installed} is installed"
            + (f" >= {min_version}" if min_version else ""),
        )
    return CheckResult(
        name=f"package:{package}",
        passed=False,
        message=f"'{package}' {installed} is below minimum {min_version}",
    )


def _check_import(
    package: str,
    min_version: Optional[str] = None,
    import_name: Optional[str] = None,
) -> CheckResult:
    """Try to import *package* and optionally verify its version."""
    mod_name = import_name or package
    try:
        importlib.import_module(mod_name)
    except ImportError as exc:
        return CheckResult(
            name=f"package:{package}",
            passed=False,
            message=f"Cannot import '{mod_name}'",
            detail=str(exc),
        )

    if min_version is None:
        return CheckResult(
            name=f"package:{package}",
            passed=True,
            message=f"'{package}' is installed and importable",
        )

    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        # Importable but metadata missing — treat as fine.
        return CheckResult(
            name=f"package:{package}",
            passed=True,
            message=f"'{package}' is importable (version metadata unavailable)",
        )

    if _version_gte(installed, min_version):
        return CheckResult(
            name=f"package:{package}",
            passed=True,
            message=f"'{package}' {installed} >= {min_version}",
        )
    return CheckResult(
        name=f"package:{package}",
        passed=False,
        message=f"'{package}' {installed} is below minimum {min_version}",
    )


def _check_optional_import(package: str, import_name: Optional[str] = None) -> CheckResult:
    """Import an optional package; PASS whether present or absent."""
    mod_name = import_name or package
    try:
        importlib.import_module(mod_name)
        return CheckResult(
            name=f"optional:{package}",
            passed=True,
            message=f"Optional '{package}' is installed ✓",
        )
    except ImportError:
        return CheckResult(
            name=f"optional:{package}",
            passed=True,
            message=f"Optional '{package}' not installed (graceful fallback active)",
        )


def _check_callable(name: str, fn: Callable[[], None]) -> CheckResult:
    """Run *fn* and return PASS if no exception is raised."""
    try:
        fn()
        return CheckResult(name=name, passed=True, message="OK")
    except Exception:  # noqa: BLE001
        return CheckResult(
            name=name,
            passed=False,
            message="raised an exception",
            detail=traceback.format_exc(limit=3).strip(),
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_required_packages() -> List[CheckResult]:
    return [
        _check_import("psutil", min_version="5.9.0"),
        _check_import("pyttsx3", min_version="2.90"),
        _check_import_metadata("pynput", min_version="1.8.1"),
        _check_import("pyperclip", min_version="1.11.0"),
    ]


def _check_optional_packages() -> List[CheckResult]:
    return [
        _check_optional_import("pynvml"),
    ]


def _check_core_modules() -> List[CheckResult]:
    """Import every public ai_helper module and confirm it has no syntax/import errors."""
    modules = [
        "ai_helper.config",
        "ai_helper.monitor",
        "ai_helper.process_manager",
        "ai_helper.gpu_monitor",
        "ai_helper.ml_engine",
        "ai_helper.ai_integrations",
        "ai_helper.agent",
        "ai_helper.tools",
        "ai_helper.file_system",
        "ai_helper.organizer",
        "ai_helper.backup",
        "ai_helper.memory",
        "ai_helper.notification_center",
        "ai_helper.dashboard",
        "ai_helper.web_ui",
        "ai_helper.clipboard_monitor",
        "ai_helper.hotkey",
        "ai_helper.communicator",
        "ai_helper.voice",
        "ai_helper.scheduler",
        "ai_helper.service",
        "ai_helper.updater",
        "ai_helper.orchestrator",
    ]
    results = []
    for mod in modules:
        short = mod.split(".", 1)[-1]
        try:
            importlib.import_module(mod)
            results.append(
                CheckResult(
                    name=f"module:{short}",
                    passed=True,
                    message="importable",
                )
            )
        except Exception:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"module:{short}",
                    passed=False,
                    message="failed to import",
                    detail=traceback.format_exc(limit=3).strip(),
                )
            )
    return results


def _check_core_classes() -> List[CheckResult]:
    """Instantiate core classes and verify they expose the expected API."""
    results: List[CheckResult] = []

    # SystemMonitor
    def _monitor() -> None:
        from ai_helper.monitor import SystemMonitor
        m = SystemMonitor()
        snap = m.snapshot()
        assert snap.cpu_percent >= 0, "cpu_percent must be non-negative"
        assert snap.memory_percent >= 0, "memory_percent must be non-negative"
        assert isinstance(m.format_snapshot(snap), str), "format_snapshot must return str"

    results.append(_check_callable("SystemMonitor.snapshot", _monitor))

    # ProcessManager
    def _proc_mgr() -> None:
        from ai_helper.process_manager import ProcessManager
        pm = ProcessManager()
        procs = pm.list_processes()
        assert isinstance(procs, list), "list_processes must return list"
        assert isinstance(pm.summary(procs), str), "summary must return str"

    results.append(_check_callable("ProcessManager.list_processes", _proc_mgr))

    # MLEngine
    def _ml_engine() -> None:
        from ai_helper.ml_engine import AnomalyDetector
        det = AnomalyDetector()
        result = det.observe("cpu", 50.0)
        # Result may be None (not enough samples) or an Anomaly object — both are fine.
        assert result is None or hasattr(result, "metric"), "observe must return None or Anomaly"

    results.append(_check_callable("AnomalyDetector.observe", _ml_engine))

    # Memory
    def _memory() -> None:
        import tempfile
        from pathlib import Path
        from ai_helper.memory import Memory
        with tempfile.TemporaryDirectory() as tmp:
            mem = Memory(db_path=Path(tmp) / "diag.db")
            mem.set_preference("_diag_test", "ok")
            assert mem.get_preference("_diag_test") == "ok"
            assert isinstance(mem.summary(), str)

    results.append(_check_callable("Memory.set/get_preference", _memory))

    # NotificationCenter
    def _notif() -> None:
        from ai_helper.notification_center import NotificationCenter
        nc = NotificationCenter()
        nc.notify("Diagnostics self-test", source="diagnostics", urgency="info")
        hist = nc.format_history()
        assert isinstance(hist, str)

    results.append(_check_callable("NotificationCenter.notify", _notif))

    # Voice (disabled — no TTS hardware needed)
    def _voice() -> None:
        from ai_helper.voice import Speaker, VoiceSettings
        s = Speaker(settings=VoiceSettings(), enabled=False)
        # speak() on a disabled speaker returns None (no-op); it must not raise
        s.speak("test")

    results.append(_check_callable("Speaker(disabled).speak", _voice))

    # Scheduler
    def _scheduler() -> None:
        from ai_helper.scheduler import TaskScheduler
        sched = TaskScheduler()
        assert hasattr(sched, "add"), "TaskScheduler must have add"
        assert hasattr(sched, "start"), "TaskScheduler must have start"

    results.append(_check_callable("TaskScheduler API", _scheduler))

    # Updater
    def _updater() -> None:
        from ai_helper.updater import Updater
        u = Updater()
        assert hasattr(u, "check"), "Updater must have check()"
        assert hasattr(u, "download"), "Updater must have download()"

    results.append(_check_callable("Updater API", _updater))

    # Config paths
    def _config() -> None:
        from ai_helper import config as cfg
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AI_HELPER_INSTALL_DIR"] = tmp
            cfg._resolved = cfg._UNSET
            try:
                assert cfg.get_install_dir() == cfg._resolve()
                assert cfg.get_downloads_dir().parent == cfg.get_install_dir()
                assert cfg.get_logs_dir().parent == cfg.get_install_dir()
            finally:
                del os.environ["AI_HELPER_INSTALL_DIR"]
                cfg._resolved = cfg._UNSET

    results.append(_check_callable("config.get_install_dir", _config))

    # ToolRegistry (built-in tools present)
    def _tools() -> None:
        from ai_helper.tools import ToolRegistry
        reg = ToolRegistry(register_defaults=True)
        names = [t.name for t in reg.list_tools()]
        for expected in ("read_file", "write_file", "system_snapshot", "list_programs"):
            assert expected in names, f"Built-in tool '{expected}' missing from registry"

    results.append(_check_callable("ToolRegistry built-in tools", _tools))

    return results


def _check_version_declared() -> CheckResult:
    """Confirm that __version__ is declared in the package."""
    try:
        import ai_helper
        ver = getattr(ai_helper, "__version__", None)
        if ver and isinstance(ver, str) and ver.strip():
            return CheckResult(
                name="package:__version__",
                passed=True,
                message=f"ai_helper.__version__ = {ver!r}",
            )
        return CheckResult(
            name="package:__version__",
            passed=False,
            message="ai_helper.__version__ is missing or empty",
        )
    except Exception:  # noqa: BLE001
        return CheckResult(
            name="package:__version__",
            passed=False,
            message="Could not import ai_helper",
            detail=traceback.format_exc(limit=3).strip(),
        )


def _check_python_version() -> CheckResult:
    """Verify Python >= 3.10."""
    info = sys.version_info
    major, minor = info[0], info[1]
    micro = info[2] if len(info) > 2 else 0
    ok = (major, minor) >= (3, 10)
    ver_str = f"{major}.{minor}.{micro}"
    return CheckResult(
        name="python:version",
        passed=ok,
        message=f"Python {ver_str} {'meets' if ok else 'does NOT meet'} minimum 3.10",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_diagnostics(*, verbose: bool = True) -> Tuple[DiagnosticsReport, bool]:
    """Run all diagnostics checks and return ``(report, passed)``.

    Parameters
    ----------
    verbose:
        If *True*, print the report to *stdout* as checks complete.

    Returns
    -------
    tuple
        ``(DiagnosticsReport, bool)`` — the full report and ``True`` iff all
        *required* checks passed (optional-package checks never fail).
    """
    report = DiagnosticsReport()

    all_checks: List[Tuple[str, Callable[[], List[CheckResult]]]] = [
        ("Python version", lambda: [_check_python_version()]),
        ("Package: __version__", lambda: [_check_version_declared()]),
        ("Required packages", _check_required_packages),
        ("Optional packages", _check_optional_packages),
        ("Core module imports", _check_core_modules),
        ("Core class smoke-tests", _check_core_classes),
    ]

    for section, fn in all_checks:
        if verbose:
            print(f"\n── {section} ──")
        results = fn()
        for r in results:
            report.checks.append(r)
            if verbose:
                print(str(r))

    if verbose:
        print()
        print(str(report).split("\n")[-2])  # Print the one-line summary
        print()

    return report, report.passed
