"""Orchestrator.

Wires the :class:`~ai_helper.monitor.SystemMonitor`,
:class:`~ai_helper.process_manager.ProcessManager` and
:class:`~ai_helper.communicator.Communicator` together into a single
polling loop that keeps the desktop *running smooth*, *organised* and
*communicating*.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .communicator import Communicator
from .monitor import SystemMonitor, SystemSnapshot
from .process_manager import ProcessManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates all AI Helper sub-systems.

    Parameters
    ----------
    poll_interval:
        Seconds between each monitoring cycle.  Default is 30 s.
    monitor:
        Optional pre-configured :class:`~ai_helper.monitor.SystemMonitor`.
    process_manager:
        Optional pre-configured :class:`~ai_helper.process_manager.ProcessManager`.
    communicator:
        Optional pre-configured :class:`~ai_helper.communicator.Communicator`.
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        monitor: Optional[SystemMonitor] = None,
        process_manager: Optional[ProcessManager] = None,
        communicator: Optional[Communicator] = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.monitor = monitor or SystemMonitor()
        self.process_manager = process_manager or ProcessManager()
        self.communicator = communicator or Communicator()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring loop in a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Orchestrator already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ai-helper-orchestrator", daemon=True)
        self._thread.start()
        logger.info("AI Helper orchestrator started (interval=%ss)", self.poll_interval)
        self.communicator.publish("status", "started")

    def stop(self) -> None:
        """Signal the background loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 5)
        logger.info("AI Helper orchestrator stopped")
        self.communicator.publish("status", "stopped")

    @property
    def running(self) -> bool:
        """``True`` while the background loop is active."""
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    # ------------------------------------------------------------------
    # One monitoring cycle (public so tests can call it directly)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Run a single monitoring cycle synchronously."""
        self._check_system()
        self._check_processes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error in orchestrator tick")
            self._stop_event.wait(timeout=self.poll_interval)

    def _check_system(self) -> None:
        snap: SystemSnapshot = self.monitor.snapshot()
        self.communicator.publish("snapshot", snap, source="monitor")
        logger.debug("%s", self.monitor.format_snapshot(snap))

        for alert_msg in self.monitor.alerts(snap):
            logger.warning("SYSTEM ALERT: %s", alert_msg)
            self.communicator.alert(alert_msg, source="monitor", urgency="critical")

    def _check_processes(self) -> None:
        procs = self.process_manager.list_processes()
        self.communicator.publish("processes", procs, source="process_manager")
        logger.debug("%s", self.process_manager.summary(procs))

        for proc in self.process_manager.high_cpu_processes(procs):
            msg = f"Process {proc.name!r} (PID {proc.pid}) using {proc.cpu_percent:.1f}% CPU"
            logger.warning("PROCESS ALERT: %s", msg)
            self.communicator.alert(msg, source="process_manager", urgency="normal", topic="process_alert")

        for proc in self.process_manager.high_memory_processes(procs):
            msg = (
                f"Process {proc.name!r} (PID {proc.pid}) "
                f"using {proc.memory_mb:.0f} MB memory"
            )
            logger.warning("PROCESS ALERT: %s", msg)
            self.communicator.alert(msg, source="process_manager", urgency="normal", topic="process_alert")
