"""System resource monitor.

Collects CPU, memory, disk and network statistics so the AI Helper can
keep the desktop *running smooth* and alert the user when resources are
under pressure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil


@dataclass
class DiskStats:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float


@dataclass
class NetworkStats:
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int


@dataclass
class SystemSnapshot:
    """A point-in-time snapshot of system resource usage."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disks: List[DiskStats] = field(default_factory=list)
    network: Optional[NetworkStats] = None


# Thresholds (%) above which a resource is considered stressed.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cpu": 85.0,
    "memory": 85.0,
    "disk": 90.0,
}


class SystemMonitor:
    """Polls system resources and reports snapshots and alerts.

    Parameters
    ----------
    thresholds:
        Dict with optional keys ``cpu``, ``memory``, ``disk`` specifying
        the percentage above which an alert is raised.  Values default to
        ``DEFAULT_THRESHOLDS``.
    disk_paths:
        List of filesystem paths to monitor.  Defaults to the root path.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        disk_paths: Optional[List[str]] = None,
    ) -> None:
        self.thresholds: Dict[str, float] = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.disk_paths: List[str] = disk_paths if disk_paths is not None else ["/"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Return a fresh :class:`SystemSnapshot`."""
        vm = psutil.virtual_memory()
        net = psutil.net_io_counters()

        disks: List[DiskStats] = []
        for path in self.disk_paths:
            try:
                usage = psutil.disk_usage(path)
                disks.append(
                    DiskStats(
                        path=path,
                        total_gb=round(usage.total / 1e9, 2),
                        used_gb=round(usage.used / 1e9, 2),
                        free_gb=round(usage.free / 1e9, 2),
                        percent=usage.percent,
                    )
                )
            except (PermissionError, FileNotFoundError):
                pass

        network = NetworkStats(
            bytes_sent=net.bytes_sent,
            bytes_recv=net.bytes_recv,
            packets_sent=net.packets_sent,
            packets_recv=net.packets_recv,
        ) if net else None

        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=vm.percent,
            memory_used_mb=round(vm.used / 1e6, 1),
            memory_total_mb=round(vm.total / 1e6, 1),
            disks=disks,
            network=network,
        )

    def alerts(self, snap: Optional[SystemSnapshot] = None) -> List[str]:
        """Return a list of human-readable alert strings for stressed resources.

        Parameters
        ----------
        snap:
            Snapshot to evaluate.  If *None*, a fresh snapshot is taken.
        """
        if snap is None:
            snap = self.snapshot()

        messages: List[str] = []

        if snap.cpu_percent >= self.thresholds["cpu"]:
            messages.append(
                f"High CPU usage: {snap.cpu_percent:.1f}% "
                f"(threshold {self.thresholds['cpu']:.0f}%)"
            )

        if snap.memory_percent >= self.thresholds["memory"]:
            messages.append(
                f"High memory usage: {snap.memory_percent:.1f}% "
                f"({snap.memory_used_mb:.0f} MB / {snap.memory_total_mb:.0f} MB) "
                f"(threshold {self.thresholds['memory']:.0f}%)"
            )

        for disk in snap.disks:
            if disk.percent >= self.thresholds["disk"]:
                messages.append(
                    f"Low disk space on {disk.path}: {disk.percent:.1f}% used "
                    f"({disk.free_gb:.2f} GB free) "
                    f"(threshold {self.thresholds['disk']:.0f}%)"
                )

        return messages

    def format_snapshot(self, snap: Optional[SystemSnapshot] = None) -> str:
        """Return a multi-line human-readable summary of the current snapshot."""
        if snap is None:
            snap = self.snapshot()

        lines = [
            f"=== System Snapshot ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap.timestamp))}) ===",
            f"  CPU:    {snap.cpu_percent:5.1f}%",
            f"  Memory: {snap.memory_percent:5.1f}%  ({snap.memory_used_mb:.0f} MB / {snap.memory_total_mb:.0f} MB)",
        ]
        for disk in snap.disks:
            lines.append(
                f"  Disk [{disk.path}]:  {disk.percent:5.1f}%  "
                f"({disk.free_gb:.2f} GB free of {disk.total_gb:.2f} GB)"
            )
        if snap.network:
            lines.append(
                f"  Network: ↑ {snap.network.bytes_sent / 1e6:.1f} MB sent  "
                f"↓ {snap.network.bytes_recv / 1e6:.1f} MB recv"
            )
        return "\n".join(lines)
