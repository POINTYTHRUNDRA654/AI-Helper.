"""Auto-start / always-running service installer.

Registers AI Helper as a persistent background service so it starts
automatically when the computer boots and is automatically restarted
if it ever crashes.

Supported platforms
-------------------
* **Windows** – Windows Task Scheduler task (``schtasks``).
  Triggers at every user logon; restarts on failure every 60 s.
* **Linux** – systemd *user* service
  (``~/.config/systemd/user/ai-helper.service``).
  ``Restart=always`` with a 5 s delay between restarts.
* **macOS** – launchd *user agent*
  (``~/Library/LaunchAgents/com.ai-helper.plist``).
  ``KeepAlive = true`` so launchd relaunches it immediately on exit.

Usage
-----
Install::

    python -m ai_helper --install-service

Uninstall::

    python -m ai_helper --uninstall-service

Or call the API directly::

    from ai_helper.service import ServiceManager
    mgr = ServiceManager()
    mgr.install()
    mgr.uninstall()
    mgr.status()
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()
_SERVICE_NAME = "ai-helper"
_DISPLAY_NAME = "AI Helper"
_DESCRIPTION = "AI Helper – keeps the desktop running smooth, organised and communicating."


@dataclass
class ServiceStatus:
    installed: bool
    running: bool
    platform: str
    details: str

    def __str__(self) -> str:
        state = "running" if self.running else ("installed" if self.installed else "not installed")
        return f"[{self.platform}] AI Helper service: {state}\n  {self.details}"


class ServiceManager:
    """Platform-aware installer for the AI Helper background service.

    Parameters
    ----------
    python_executable:
        Path to the Python interpreter that should run AI Helper.
        Defaults to the current interpreter (``sys.executable``).
    extra_args:
        Additional command-line arguments appended to ``python -m ai_helper
        --daemon``.  Example: ``["--voice", "--interval", "60"]``.
    install_dir:
        The AI Helper install directory used for log paths.  Defaults to the
        value from :mod:`ai_helper.config`.
    """

    def __init__(
        self,
        python_executable: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        install_dir: Optional[Path] = None,
    ) -> None:
        self.python = python_executable or sys.executable
        self.extra_args = extra_args or []
        from . import config as _cfg  # lazy to avoid circular at package init
        self.install_dir = install_dir or _cfg.get_install_dir()
        self.log_dir = self.install_dir / "Logs"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self) -> bool:
        """Install and enable the AI Helper service/startup task.

        Returns ``True`` on success, ``False`` on failure.
        """
        logger.info("Installing AI Helper service on %s…", _SYSTEM)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if _SYSTEM == "Windows":
            return self._install_windows()
        if _SYSTEM == "Linux":
            return self._install_linux()
        if _SYSTEM == "Darwin":
            return self._install_macos()

        logger.error("Auto-start not supported on %s", _SYSTEM)
        return False

    def uninstall(self) -> bool:
        """Remove the AI Helper service/startup task.

        Returns ``True`` on success, ``False`` on failure.
        """
        logger.info("Uninstalling AI Helper service on %s…", _SYSTEM)

        if _SYSTEM == "Windows":
            return self._uninstall_windows()
        if _SYSTEM == "Linux":
            return self._uninstall_linux()
        if _SYSTEM == "Darwin":
            return self._uninstall_macos()

        logger.error("Auto-start not supported on %s", _SYSTEM)
        return False

    def status(self) -> ServiceStatus:
        """Return the current installation and running status."""
        if _SYSTEM == "Windows":
            return self._status_windows()
        if _SYSTEM == "Linux":
            return self._status_linux()
        if _SYSTEM == "Darwin":
            return self._status_macos()
        return ServiceStatus(installed=False, running=False, platform=_SYSTEM,
                             details="Platform not supported.")

    # ------------------------------------------------------------------
    # Windows – Task Scheduler
    # ------------------------------------------------------------------

    def _cmd(self) -> str:
        """Return the full command string for the daemon."""
        parts = [self.python, "-m", "ai_helper", "--daemon"] + self.extra_args
        return " ".join(f'"{p}"' if " " in str(p) else str(p) for p in parts)

    def _install_windows(self) -> bool:
        stdout_log = self.log_dir / "ai-helper-stdout.log"
        cmd = self._cmd()
        # Wrap in a PowerShell script so we can redirect output to a log file.
        ps_wrapper = (
            f"Start-Process -NoNewWindow -FilePath 'cmd.exe' "
            f"-ArgumentList '/c {cmd} >> \"{stdout_log}\" 2>&1'"
        )
        schtasks_cmd = [
            "schtasks", "/Create", "/F",
            "/TN", _SERVICE_NAME,
            "/TR", f"powershell -NonInteractive -WindowStyle Hidden -Command \"{ps_wrapper}\"",
            "/SC", "ONLOGON",
            "/RL", "HIGHEST",
            "/DELAY", "0001:00",  # 1 minute after logon
        ]
        result = subprocess.run(schtasks_cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode == 0:
            logger.info("Task Scheduler task %r created.", _SERVICE_NAME)
            # Also set it to restart on failure via XML – best effort
            self._windows_set_restart_on_failure()
            return True
        logger.error("schtasks failed: %s", result.stderr.strip())
        return False

    def _windows_set_restart_on_failure(self) -> None:
        """Export, patch and re-import the task XML to add restart-on-failure."""
        try:
            export = subprocess.run(
                ["schtasks", "/Query", "/TN", _SERVICE_NAME, "/XML"],
                capture_output=True, text=True,
            )  # noqa: S603
            if export.returncode != 0:
                return
            xml = export.stdout
            # Inject RestartOnFailure settings if not already present
            if "RestartOnFailure" not in xml:
                restart_xml = (
                    "<RestartOnFailure>"
                    "<Interval>PT1M</Interval>"
                    "<Count>99</Count>"
                    "</RestartOnFailure>"
                )
                xml = xml.replace("</Settings>", f"{restart_xml}</Settings>", 1)
            xml_path = self.log_dir / "ai-helper-task.xml"
            xml_path.write_text(xml, encoding="utf-8")
            subprocess.run(
                ["schtasks", "/Create", "/F", "/TN", _SERVICE_NAME, "/XML", str(xml_path)],
                capture_output=True,
            )  # noqa: S603
        except Exception:  # noqa: BLE001
            logger.debug("Could not patch task for restart-on-failure", exc_info=True)

    def _uninstall_windows(self) -> bool:
        result = subprocess.run(
            ["schtasks", "/Delete", "/F", "/TN", _SERVICE_NAME],
            capture_output=True, text=True,
        )  # noqa: S603
        if result.returncode == 0:
            logger.info("Task Scheduler task %r deleted.", _SERVICE_NAME)
            return True
        logger.error("schtasks /Delete failed: %s", result.stderr.strip())
        return False

    def _status_windows(self) -> ServiceStatus:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", _SERVICE_NAME, "/FO", "LIST"],
            capture_output=True, text=True,
        )  # noqa: S603
        installed = result.returncode == 0
        running = installed and "Running" in result.stdout
        return ServiceStatus(
            installed=installed,
            running=running,
            platform="Windows (Task Scheduler)",
            details=result.stdout.strip() if installed else "Task not found.",
        )

    # ------------------------------------------------------------------
    # Linux – systemd user service
    # ------------------------------------------------------------------

    @property
    def _systemd_service_dir(self) -> Path:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(xdg) / "systemd" / "user"

    @property
    def _systemd_service_file(self) -> Path:
        return self._systemd_service_dir / f"{_SERVICE_NAME}.service"

    def _install_linux(self) -> bool:
        stdout_log = self.log_dir / "ai-helper-stdout.log"
        stderr_log = self.log_dir / "ai-helper-stderr.log"
        cmd_parts = [self.python, "-m", "ai_helper", "--daemon"] + self.extra_args
        exec_start = " ".join(str(p) for p in cmd_parts)

        unit = (
            f"[Unit]\n"
            f"Description={_DESCRIPTION}\n"
            f"After=network.target\n"
            f"\n"
            f"[Service]\n"
            f"Type=simple\n"
            f"ExecStart={exec_start}\n"
            f"StandardOutput=append:{stdout_log}\n"
            f"StandardError=append:{stderr_log}\n"
            f"Restart=always\n"
            f"RestartSec=5s\n"
            f"\n"
            f"[Install]\n"
            f"WantedBy=default.target\n"
        )

        self._systemd_service_dir.mkdir(parents=True, exist_ok=True)
        self._systemd_service_file.write_text(unit, encoding="utf-8")
        logger.info("Wrote systemd unit → %s", self._systemd_service_file)

        if shutil.which("systemctl"):
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            result = subprocess.run(
                ["systemctl", "--user", "enable", "--now", f"{_SERVICE_NAME}.service"],
                capture_output=True, text=True,
            )  # noqa: S603
            if result.returncode == 0:
                logger.info("systemd service enabled and started.")
                return True
            logger.warning("systemctl enable failed: %s", result.stderr.strip())
            logger.info("Service file written; enable manually with:\n"
                        "  systemctl --user enable --now %s", _SERVICE_NAME)
        else:
            logger.info("systemctl not found; enable manually with:\n"
                        "  systemctl --user enable --now %s", _SERVICE_NAME)
        return True  # File written successfully even if systemctl unavailable

    def _uninstall_linux(self) -> bool:
        if shutil.which("systemctl"):
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", f"{_SERVICE_NAME}.service"],
                check=False, capture_output=True,
            )
        if self._systemd_service_file.exists():
            self._systemd_service_file.unlink()
            logger.info("Removed %s", self._systemd_service_file)
        if shutil.which("systemctl"):
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        return True

    def _status_linux(self) -> ServiceStatus:
        installed = self._systemd_service_file.exists()
        running = False
        details = str(self._systemd_service_file) if installed else "Unit file not found."
        if installed and shutil.which("systemctl"):
            r = subprocess.run(
                ["systemctl", "--user", "is-active", f"{_SERVICE_NAME}.service"],
                capture_output=True, text=True,
            )
            running = r.stdout.strip() == "active"
            details = r.stdout.strip()
        return ServiceStatus(installed=installed, running=running,
                             platform="Linux (systemd user service)", details=details)

    # ------------------------------------------------------------------
    # macOS – launchd user agent
    # ------------------------------------------------------------------

    @property
    def _launchd_plist_dir(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents"

    @property
    def _launchd_plist_file(self) -> Path:
        return self._launchd_plist_dir / f"com.{_SERVICE_NAME}.plist"

    def _install_macos(self) -> bool:
        stdout_log = self.log_dir / "ai-helper-stdout.log"
        stderr_log = self.log_dir / "ai-helper-stderr.log"
        cmd_parts = [self.python, "-m", "ai_helper", "--daemon"] + self.extra_args
        program_args = "\n".join(f"        <string>{p}</string>" for p in cmd_parts)

        plist = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
            ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            '<dict>\n'
            f'    <key>Label</key>\n'
            f'    <string>com.{_SERVICE_NAME}</string>\n'
            f'    <key>ProgramArguments</key>\n'
            f'    <array>\n'
            f'{program_args}\n'
            f'    </array>\n'
            f'    <key>KeepAlive</key>\n'
            f'    <true/>\n'
            f'    <key>RunAtLoad</key>\n'
            f'    <true/>\n'
            f'    <key>StandardOutPath</key>\n'
            f'    <string>{stdout_log}</string>\n'
            f'    <key>StandardErrorPath</key>\n'
            f'    <string>{stderr_log}</string>\n'
            f'    <key>ThrottleInterval</key>\n'
            f'    <integer>5</integer>\n'
            '</dict>\n'
            '</plist>\n'
        )

        self._launchd_plist_dir.mkdir(parents=True, exist_ok=True)
        self._launchd_plist_file.write_text(plist, encoding="utf-8")
        logger.info("Wrote launchd plist → %s", self._launchd_plist_file)

        if shutil.which("launchctl"):
            result = subprocess.run(
                ["launchctl", "load", "-w", str(self._launchd_plist_file)],
                capture_output=True, text=True,
            )  # noqa: S603
            if result.returncode == 0:
                logger.info("launchd agent loaded.")
                return True
            logger.warning("launchctl load failed: %s", result.stderr.strip())
        return True  # Plist written even if launchctl not run

    def _uninstall_macos(self) -> bool:
        if shutil.which("launchctl") and self._launchd_plist_file.exists():
            subprocess.run(
                ["launchctl", "unload", "-w", str(self._launchd_plist_file)],
                check=False, capture_output=True,
            )
        if self._launchd_plist_file.exists():
            self._launchd_plist_file.unlink()
            logger.info("Removed %s", self._launchd_plist_file)
        return True

    def _status_macos(self) -> ServiceStatus:
        installed = self._launchd_plist_file.exists()
        running = False
        details = str(self._launchd_plist_file) if installed else "Plist not found."
        if shutil.which("launchctl"):
            r = subprocess.run(
                ["launchctl", "list", f"com.{_SERVICE_NAME}"],
                capture_output=True, text=True,
            )
            running = r.returncode == 0
            if running:
                details = r.stdout.strip()
        return ServiceStatus(installed=installed, running=running,
                             platform="macOS (launchd user agent)", details=details)
