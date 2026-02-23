"""Task scheduler.

A lightweight, thread-safe interval-based task scheduler so AI Helper
can *schedule* recurring work (e.g. monitoring ticks, periodic organise
runs, report generation) without relying on external cron or APScheduler.

Each :class:`Task` specifies a callable and an interval; the
:class:`TaskScheduler` runs them in a single daemon thread, catching and
logging exceptions so one bad task never crashes the whole application.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A named recurring task.

    Parameters
    ----------
    name:
        Unique human-readable name.
    func:
        Zero-argument callable to invoke on each tick.
    interval:
        Seconds between executions.
    enabled:
        Set to ``False`` to pause without removing from the scheduler.
    """

    name: str
    func: Callable[[], None]
    interval: float
    enabled: bool = True
    run_count: int = field(default=0, init=False)
    error_count: int = field(default=0, init=False)
    last_run: Optional[float] = field(default=None, init=False)
    last_error: Optional[str] = field(default=None, init=False)
    _next_run: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._next_run = time.monotonic() + self.interval

    def is_due(self) -> bool:
        return self.enabled and time.monotonic() >= self._next_run

    def run(self) -> None:
        try:
            self.func()
            self.run_count += 1
            self.last_run = time.time()
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.error_count += 1
            self.last_error = str(exc)
            logger.exception("Task %r raised an error", self.name)
        finally:
            self._next_run = time.monotonic() + self.interval


@dataclass
class TaskStatus:
    name: str
    interval: float
    enabled: bool
    run_count: int
    error_count: int
    last_run: Optional[float]
    last_error: Optional[str]

    def __str__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        last = (
            time.strftime("%H:%M:%S", time.localtime(self.last_run))
            if self.last_run
            else "never"
        )
        err = f"  last_error={self.last_error!r}" if self.last_error else ""
        return (
            f"[{self.name}] {status}  interval={self.interval:.0f}s  "
            f"runs={self.run_count}  errors={self.error_count}  last_run={last}{err}"
        )


class TaskScheduler:
    """Run registered :class:`Task` objects on their configured intervals.

    The scheduler runs in a single background daemon thread, sleeping for
    at most *resolution* seconds between checks so it stays responsive
    to newly added tasks or ``stop()`` calls.

    Parameters
    ----------
    resolution:
        Seconds between internal schedule checks (default: 1 s).
    """

    def __init__(self, resolution: float = 1.0) -> None:
        self.resolution = resolution
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        func: Callable[[], None],
        interval: float,
        *,
        enabled: bool = True,
        replace: bool = False,
    ) -> Task:
        """Register a new recurring task.

        Parameters
        ----------
        name:
            Unique task name.  Raises :exc:`ValueError` if already registered
            unless *replace* is ``True``.
        func:
            Zero-argument callable.
        interval:
            Seconds between executions.
        enabled:
            Start enabled (default ``True``).
        replace:
            If ``True``, silently replace an existing task with the same name.
        """
        with self._lock:
            if name in self._tasks and not replace:
                raise ValueError(f"Task {name!r} already registered; use replace=True to overwrite")
            task = Task(name=name, func=func, interval=interval, enabled=enabled)
            self._tasks[name] = task
            logger.debug("Registered task %r (interval=%ss)", name, interval)
            return task

    def remove(self, name: str) -> bool:
        """Unregister a task by name.  Returns ``True`` if it existed."""
        with self._lock:
            if name in self._tasks:
                del self._tasks[name]
                logger.debug("Removed task %r", name)
                return True
            return False

    def enable(self, name: str) -> None:
        """Enable a paused task."""
        with self._lock:
            self._tasks[name].enabled = True

    def disable(self, name: str) -> None:
        """Pause a task without removing it."""
        with self._lock:
            self._tasks[name].enabled = False

    def get(self, name: str) -> Optional[Task]:
        """Return a task by name, or *None*."""
        with self._lock:
            return self._tasks.get(name)

    def status(self) -> List[TaskStatus]:
        """Return a :class:`TaskStatus` snapshot for every registered task."""
        with self._lock:
            return [
                TaskStatus(
                    name=t.name,
                    interval=t.interval,
                    enabled=t.enabled,
                    run_count=t.run_count,
                    error_count=t.error_count,
                    last_run=t.last_run,
                    last_error=t.last_error,
                )
                for t in self._tasks.values()
            ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ai-helper-scheduler", daemon=True
        )
        self._thread.start()
        logger.info("Task scheduler started")

    def stop(self) -> None:
        """Stop the scheduler and wait for the background thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.resolution * 3)
        logger.info("Task scheduler stopped")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    # ------------------------------------------------------------------
    # Run once (useful for testing)
    # ------------------------------------------------------------------

    def run_due(self) -> int:
        """Execute all currently due tasks synchronously.  Returns count run."""
        due: List[Task] = []
        with self._lock:
            due = [t for t in self._tasks.values() if t.is_due()]
        for task in due:
            task.run()
        return len(due)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_due()
            self._stop_event.wait(timeout=self.resolution)
