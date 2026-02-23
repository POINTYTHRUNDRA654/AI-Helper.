"""Process manager.

Lists, monitors and optionally terminates OS processes so the AI Helper
can keep the desktop *organised* and identify misbehaving programs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Lightweight snapshot of a single process."""

    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    num_threads: int


class ProcessManager:
    """Manage and monitor running OS processes.

    Parameters
    ----------
    cpu_threshold:
        Percentage above which a single process is flagged as a CPU hog.
    memory_threshold_mb:
        MB above which a single process is flagged as a memory hog.
    """

    def __init__(
        self,
        cpu_threshold: float = 50.0,
        memory_threshold_mb: float = 500.0,
    ) -> None:
        self.cpu_threshold = cpu_threshold
        self.memory_threshold_mb = memory_threshold_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_processes(self) -> List[ProcessInfo]:
        """Return a list of :class:`ProcessInfo` for all running processes."""
        result: List[ProcessInfo] = []
        for proc in psutil.process_iter(
            ["pid", "name", "status", "cpu_percent", "memory_info", "num_threads"]
        ):
            try:
                info = proc.info
                mem_mb = (
                    round(info["memory_info"].rss / 1e6, 1)
                    if info.get("memory_info")
                    else 0.0
                )
                result.append(
                    ProcessInfo(
                        pid=info["pid"],
                        name=info["name"] or "<unknown>",
                        status=info["status"] or "unknown",
                        cpu_percent=info.get("cpu_percent") or 0.0,
                        memory_mb=mem_mb,
                        num_threads=info.get("num_threads") or 0,
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return result

    def find_by_name(self, name: str) -> List[ProcessInfo]:
        """Return all processes whose name contains *name* (case-insensitive)."""
        name_lower = name.lower()
        return [p for p in self.list_processes() if name_lower in p.name.lower()]

    def high_cpu_processes(
        self, processes: Optional[List[ProcessInfo]] = None
    ) -> List[ProcessInfo]:
        """Return processes exceeding :attr:`cpu_threshold`."""
        procs = processes if processes is not None else self.list_processes()
        return [p for p in procs if p.cpu_percent >= self.cpu_threshold]

    def high_memory_processes(
        self, processes: Optional[List[ProcessInfo]] = None
    ) -> List[ProcessInfo]:
        """Return processes exceeding :attr:`memory_threshold_mb`."""
        procs = processes if processes is not None else self.list_processes()
        return [p for p in procs if p.memory_mb >= self.memory_threshold_mb]

    def terminate(self, pid: int) -> bool:
        """Terminate the process with the given *pid*.

        Returns ``True`` on success, ``False`` if the process was not found
        or access was denied.
        """
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            logger.info("Terminated PID %d (%s)", pid, proc.name())
            return True
        except psutil.NoSuchProcess:
            logger.warning("PID %d not found", pid)
            return False
        except psutil.AccessDenied:
            logger.warning("Access denied terminating PID %d", pid)
            return False

    def summary(self, processes: Optional[List[ProcessInfo]] = None) -> str:
        """Return a human-readable summary of process statistics."""
        procs = processes if processes is not None else self.list_processes()
        total = len(procs)
        running = sum(1 for p in procs if p.status == "running")
        sleeping = sum(1 for p in procs if p.status == "sleeping")
        zombie = sum(1 for p in procs if p.status == "zombie")
        high_cpu = self.high_cpu_processes(procs)
        high_mem = self.high_memory_processes(procs)

        lines = [
            f"=== Process Summary (total: {total}) ===",
            f"  Running:  {running}  Sleeping: {sleeping}  Zombie: {zombie}",
        ]
        if high_cpu:
            lines.append(
                f"  High-CPU processes (>{self.cpu_threshold:.0f}%): "
                + ", ".join(f"{p.name}[{p.pid}]({p.cpu_percent:.1f}%)" for p in high_cpu)
            )
        if high_mem:
            lines.append(
                f"  High-Memory processes (>{self.memory_threshold_mb:.0f} MB): "
                + ", ".join(f"{p.name}[{p.pid}]({p.memory_mb:.0f} MB)" for p in high_mem)
            )
        return "\n".join(lines)
