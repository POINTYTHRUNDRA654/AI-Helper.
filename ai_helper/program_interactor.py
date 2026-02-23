"""Program interactor.

Allows AI Helper to *interact with all programs* on the desktop:

* **Launch** any application by command.
* **Communicate** – send input to a running process and capture its output.
* **Find** running programs by name (via psutil).
* **Signal** or terminate processes.
* **List installed applications** on common platforms.

All subprocess operations have explicit timeouts and error handling so a
frozen application cannot block the AI Helper.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LaunchResult:
    command: str
    pid: Optional[int]
    success: bool
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return f"Launched {self.command!r} → PID {self.pid}"
        return f"Failed to launch {self.command!r}: {self.error}"


@dataclass
class CommunicateResult:
    command: str
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False

    def __str__(self) -> str:
        if self.timed_out:
            return f"[TIMEOUT] {self.command!r}"
        return (
            f"{self.command!r} exited {self.returncode}\n"
            f"  stdout: {self.stdout[:200]!r}\n"
            f"  stderr: {self.stderr[:200]!r}"
        )


@dataclass
class AppInfo:
    name: str
    path: str
    installed: bool = True

    def __str__(self) -> str:
        return f"{self.name} ({self.path})"


# ---------------------------------------------------------------------------
# Core interactor
# ---------------------------------------------------------------------------


class ProgramInteractor:
    """Launch, communicate with and control programs.

    Parameters
    ----------
    default_timeout:
        Default seconds to wait before ``communicate()`` times out.
    """

    def __init__(self, default_timeout: float = 30.0) -> None:
        self.default_timeout = default_timeout

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------

    def launch(
        self,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        detach: bool = True,
    ) -> LaunchResult:
        """Launch *command* as a new process.

        Parameters
        ----------
        command:
            Executable name or full path.
        args:
            Additional arguments.
        cwd:
            Working directory for the new process.
        env:
            Extra environment variables merged with the current environment.
        detach:
            When ``True`` (default), the process is launched detached so
            it keeps running after AI Helper exits.

        Returns
        -------
        LaunchResult
        """
        full_args = shlex.split(command) + (args or [])
        merged_env = {**os.environ, **(env or {})}

        kwargs: dict = {
            "cwd": cwd,
            "env": merged_env,
        }
        if detach:
            if _SYSTEM != "Windows":
                kwargs["start_new_session"] = True
            else:
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["stdin"] = subprocess.DEVNULL
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL

        try:
            proc = subprocess.Popen(full_args, **kwargs)  # noqa: S603
            logger.info("Launched %r → PID %d", command, proc.pid)
            return LaunchResult(command=command, pid=proc.pid, success=True)
        except FileNotFoundError:
            msg = f"executable not found: {full_args[0]!r}"
            logger.error("Launch failed: %s", msg)
            return LaunchResult(command=command, pid=None, success=False, error=msg)
        except OSError as exc:
            logger.error("Launch failed: %s", exc)
            return LaunchResult(command=command, pid=None, success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Communicate (run + capture)
    # ------------------------------------------------------------------

    def communicate(
        self,
        command: str,
        args: Optional[List[str]] = None,
        input_data: Optional[str] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> CommunicateResult:
        """Run *command*, optionally send *input_data* and capture output.

        Parameters
        ----------
        command:
            Executable name or full path.
        args:
            Additional arguments.
        input_data:
            Text to write to the process's stdin.
        timeout:
            Seconds before the process is killed (defaults to
            :attr:`default_timeout`).
        cwd:
            Working directory.
        """
        full_args = shlex.split(command) + (args or [])
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            result = subprocess.run(  # noqa: S603
                full_args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return CommunicateResult(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Command %r timed out after %ss", command, timeout)
            return CommunicateResult(command=command, returncode=None, stdout="", stderr="", timed_out=True)
        except FileNotFoundError:
            msg = f"executable not found: {full_args[0]!r}"
            logger.error("communicate failed: %s", msg)
            return CommunicateResult(command=command, returncode=-1, stdout="", stderr=msg)
        except OSError as exc:
            logger.error("communicate failed: %s", exc)
            return CommunicateResult(command=command, returncode=-1, stdout="", stderr=str(exc))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_running(self, name: str) -> List[psutil.Process]:
        """Return all running processes whose name contains *name* (case-insensitive)."""
        name_lower = name.lower()
        found: List[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] and name_lower in proc.info["name"].lower():
                    found.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return found

    def list_installed(self) -> List[AppInfo]:
        """Return a best-effort list of installed applications.

        Uses platform-specific strategies:

        * **Linux** – scans ``PATH`` for executables.
        * **macOS** – lists ``/Applications/*.app`` bundles.
        * **Windows** – lists entries in common ``Program Files`` directories.
        """
        if _SYSTEM == "Linux":
            return self._list_installed_linux()
        if _SYSTEM == "Darwin":
            return self._list_installed_macos()
        if _SYSTEM == "Windows":
            return self._list_installed_windows()
        return []

    # ------------------------------------------------------------------
    # Signalling
    # ------------------------------------------------------------------

    def send_signal(self, pid: int, sig: int = signal.SIGTERM) -> bool:
        """Send *sig* to the process with *pid*.  Returns ``True`` on success."""
        try:
            os.kill(pid, sig)
            logger.info("Sent signal %d to PID %d", sig, pid)
            return True
        except ProcessLookupError:
            logger.warning("PID %d not found", pid)
            return False
        except PermissionError:
            logger.warning("Permission denied sending signal to PID %d", pid)
            return False

    def terminate(self, pid: int) -> bool:
        """Terminate (SIGTERM) a process by PID."""
        return self.send_signal(pid, signal.SIGTERM)

    def kill(self, pid: int) -> bool:
        """Forcefully kill (SIGKILL) a process by PID."""
        sig = signal.SIGKILL if _SYSTEM != "Windows" else signal.SIGTERM
        return self.send_signal(pid, sig)

    # ------------------------------------------------------------------
    # Platform helpers
    # ------------------------------------------------------------------

    def _list_installed_linux(self) -> List[AppInfo]:
        apps: List[AppInfo] = []
        seen: set = set()
        for dir_str in os.environ.get("PATH", "").split(os.pathsep):
            dir_path = Path(dir_str)
            if not dir_path.is_dir():
                continue
            try:
                for entry in dir_path.iterdir():
                    if entry.name in seen:
                        continue
                    if entry.is_file() and os.access(entry, os.X_OK):
                        seen.add(entry.name)
                        apps.append(AppInfo(name=entry.name, path=str(entry)))
            except PermissionError:
                continue
        return sorted(apps, key=lambda a: a.name.lower())

    def _list_installed_macos(self) -> List[AppInfo]:
        apps_dir = Path("/Applications")
        apps: List[AppInfo] = []
        if apps_dir.is_dir():
            for entry in apps_dir.iterdir():
                if entry.suffix == ".app":
                    apps.append(AppInfo(name=entry.stem, path=str(entry)))
        return sorted(apps, key=lambda a: a.name.lower())

    def _list_installed_windows(self) -> List[AppInfo]:
        apps: List[AppInfo] = []
        for base in [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ]:
            base_path = Path(base)
            if not base_path.is_dir():
                continue
            try:
                for exe in base_path.rglob("*.exe"):
                    apps.append(AppInfo(name=exe.stem, path=str(exe)))
            except PermissionError:
                continue
        return sorted(apps, key=lambda a: a.name.lower())
