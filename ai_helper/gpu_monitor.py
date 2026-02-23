"""NVIDIA GPU monitor.

Tracks GPU health, VRAM usage, temperature, utilisation and per-process
GPU memory so AI Helper can keep an eye on heavy AI workloads.

Two backends are tried in order:

1. **pynvml** – the official Python binding for NVIDIA's NVML library.
   Gives the most accurate, real-time data.  Install with::

       pip install pynvml

2. **nvidia-smi CLI** – ships with every NVIDIA driver; no Python package
   required.  Used automatically when pynvml is not installed.

Both paths produce the same :class:`GpuSnapshot` objects so the rest of
AI Helper never needs to know which backend is active.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GpuProcessInfo:
    """A single process consuming GPU memory."""
    pid: int
    name: str
    vram_mb: float


@dataclass
class GpuSnapshot:
    """Point-in-time snapshot of one GPU."""
    index: int
    name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_free_mb: float
    vram_percent: float
    temperature_c: float
    utilization_percent: float
    power_draw_w: float
    power_limit_w: float
    processes: List[GpuProcessInfo] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def vram_used_gb(self) -> float:
        return round(self.vram_used_mb / 1024, 2)

    @property
    def vram_total_gb(self) -> float:
        return round(self.vram_total_mb / 1024, 2)

    def __str__(self) -> str:
        return (
            f"GPU {self.index} [{self.name}]  "
            f"VRAM {self.vram_used_gb:.1f}/{self.vram_total_gb:.1f} GB "
            f"({self.vram_percent:.0f}%)  "
            f"Temp {self.temperature_c:.0f}°C  "
            f"Util {self.utilization_percent:.0f}%  "
            f"Power {self.power_draw_w:.0f}/{self.power_limit_w:.0f} W"
        )


# Default alert thresholds
DEFAULT_GPU_THRESHOLDS: Dict[str, float] = {
    "vram_percent": 90.0,
    "temperature_c": 85.0,
    "utilization_percent": 95.0,
}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _try_pynvml():
    """Return the pynvml module, or None if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]
        pynvml.nvmlInit()
        return pynvml
    except Exception:  # noqa: BLE001
        return None


def _smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------


class GpuMonitor:
    """Monitor NVIDIA GPUs via pynvml or the nvidia-smi CLI.

    Parameters
    ----------
    thresholds:
        Dict with optional keys ``vram_percent``, ``temperature_c`` and
        ``utilization_percent`` specifying alert levels.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds: Dict[str, float] = {
            **DEFAULT_GPU_THRESHOLDS, **(thresholds or {})
        }
        self._nvml = _try_pynvml()
        if self._nvml:
            logger.debug("GPU monitor: using pynvml backend")
        elif _smi_available():
            logger.debug("GPU monitor: using nvidia-smi CLI backend")
        else:
            logger.info("GPU monitor: no NVIDIA GPU or driver detected")

    @property
    def available(self) -> bool:
        """``True`` when at least one NVIDIA GPU was found."""
        return self._nvml is not None or _smi_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshots(self) -> List[GpuSnapshot]:
        """Return a :class:`GpuSnapshot` for every installed GPU."""
        if self._nvml:
            return self._snapshots_nvml()
        if _smi_available():
            return self._snapshots_smi()
        return []

    def alerts(self, snaps: Optional[List[GpuSnapshot]] = None) -> List[str]:
        """Return human-readable alert strings for any GPU exceeding thresholds."""
        if snaps is None:
            snaps = self.snapshots()
        messages: List[str] = []
        for snap in snaps:
            if snap.vram_percent >= self.thresholds["vram_percent"]:
                messages.append(
                    f"GPU {snap.index} [{snap.name}] high VRAM: "
                    f"{snap.vram_percent:.0f}% used "
                    f"({snap.vram_used_gb:.1f}/{snap.vram_total_gb:.1f} GB)"
                )
            if snap.temperature_c >= self.thresholds["temperature_c"]:
                messages.append(
                    f"GPU {snap.index} [{snap.name}] high temperature: "
                    f"{snap.temperature_c:.0f}°C"
                )
            if snap.utilization_percent >= self.thresholds["utilization_percent"]:
                messages.append(
                    f"GPU {snap.index} [{snap.name}] high utilisation: "
                    f"{snap.utilization_percent:.0f}%"
                )
        return messages

    def format_snapshots(self, snaps: Optional[List[GpuSnapshot]] = None) -> str:
        """Return a multi-line human-readable GPU summary."""
        if snaps is None:
            snaps = self.snapshots()
        if not snaps:
            return "No NVIDIA GPUs detected."
        lines = [f"=== GPU Snapshot ({time.strftime('%H:%M:%S')}) ==="]
        for snap in snaps:
            lines.append(f"  {snap}")
            for proc in snap.processes:
                lines.append(
                    f"    ↳ PID {proc.pid} [{proc.name}]  "
                    f"{proc.vram_mb:.0f} MB VRAM"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # pynvml backend
    # ------------------------------------------------------------------

    def _snapshots_nvml(self) -> List[GpuSnapshot]:
        snaps: List[GpuSnapshot] = []
        try:
            count = self._nvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
                name = self._nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temp = float(self._nvml.nvmlDeviceGetTemperature(
                        handle, self._nvml.NVML_TEMPERATURE_GPU))
                except Exception:  # noqa: BLE001
                    temp = 0.0
                try:
                    power = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit = self._nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                except Exception:  # noqa: BLE001
                    power = power_limit = 0.0

                vram_used = mem.used / 1e6
                vram_total = mem.total / 1e6
                vram_free = mem.free / 1e6
                vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0.0

                procs = self._nvml_processes(handle)
                snaps.append(GpuSnapshot(
                    index=i, name=name,
                    vram_used_mb=vram_used, vram_total_mb=vram_total,
                    vram_free_mb=vram_free, vram_percent=vram_pct,
                    temperature_c=temp,
                    utilization_percent=float(util.gpu),
                    power_draw_w=power, power_limit_w=power_limit,
                    processes=procs,
                ))
        except Exception:  # noqa: BLE001
            logger.exception("pynvml snapshot failed")
        return snaps

    def _nvml_processes(self, handle) -> List[GpuProcessInfo]:  # noqa: ANN001
        procs: List[GpuProcessInfo] = []
        try:
            import psutil
            for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                try:
                    pname = psutil.Process(p.pid).name()
                except Exception:  # noqa: BLE001
                    pname = "<unknown>"
                procs.append(GpuProcessInfo(
                    pid=p.pid, name=pname,
                    vram_mb=p.usedGpuMemory / 1e6 if p.usedGpuMemory else 0.0,
                ))
        except Exception:  # noqa: BLE001
            pass
        return procs

    # ------------------------------------------------------------------
    # nvidia-smi CLI backend
    # ------------------------------------------------------------------

    def _snapshots_smi(self) -> List[GpuSnapshot]:
        """Parse `nvidia-smi --query-gpu` CSV output into GpuSnapshot objects."""
        query = (
            "index,name,memory.used,memory.total,memory.free,"
            "temperature.gpu,utilization.gpu,power.draw,power.limit"
        )
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )  # noqa: S603
        except (subprocess.TimeoutExpired, OSError):
            return []

        if result.returncode != 0:
            return []

        snaps: List[GpuSnapshot] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                vram_used = float(parts[2])
                vram_total = float(parts[3])
                vram_free = float(parts[4])
                vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0.0
                temp = float(parts[5]) if parts[5] not in ("[N/A]", "N/A") else 0.0
                util = float(parts[6]) if parts[6] not in ("[N/A]", "N/A") else 0.0
                power = float(parts[7]) if parts[7] not in ("[N/A]", "N/A") else 0.0
                power_lim = float(parts[8]) if parts[8] not in ("[N/A]", "N/A") else 0.0
                procs = self._smi_processes(idx)
                snaps.append(GpuSnapshot(
                    index=idx, name=name,
                    vram_used_mb=vram_used, vram_total_mb=vram_total,
                    vram_free_mb=vram_free, vram_percent=vram_pct,
                    temperature_c=temp, utilization_percent=util,
                    power_draw_w=power, power_limit_w=power_lim,
                    processes=procs,
                ))
            except (ValueError, IndexError):
                continue
        return snaps

    def _smi_processes(self, gpu_index: int) -> List[GpuProcessInfo]:
        """Return per-process GPU memory via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "--query-compute-apps=pid,name,used_memory",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_index}",
                ],
                capture_output=True, text=True, timeout=10,
            )  # noqa: S603
        except (subprocess.TimeoutExpired, OSError):
            return []

        procs: List[GpuProcessInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                procs.append(GpuProcessInfo(
                    pid=int(parts[0]),
                    name=parts[1],
                    vram_mb=float(parts[2]),
                ))
            except ValueError:
                continue
        return procs
