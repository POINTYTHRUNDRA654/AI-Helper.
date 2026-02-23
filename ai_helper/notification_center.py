"""Notification center.

Manages all AI Helper alerts in one place:

* **History** — every alert is recorded with timestamp, source and urgency.
* **Deduplication** — identical alerts within a configurable cooldown window
  are suppressed so you aren't flooded with repeated warnings.
* **Throttling** — each source+topic pair can fire at most once per
  ``throttle_seconds`` (default 60 s).
* **Escalation** — if the same alert fires more than ``escalate_count`` times
  within ``escalate_window`` seconds, the urgency is automatically raised.
* **Review** — ``center.format_history()`` prints a clean alert log.

Usage
-----
::

    from ai_helper.notification_center import NotificationCenter

    nc = NotificationCenter()
    nc.notify("CPU usage is 97%", source="monitor", urgency="critical")
    nc.notify("CPU usage is 97%", source="monitor", urgency="critical")  # suppressed
    print(nc.format_history())
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class NotificationRecord:
    """One recorded notification."""
    message: str
    source: str
    urgency: str
    topic: str
    ts: float = field(default_factory=time.time)
    suppressed: bool = False
    escalated: bool = False

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))

    def __str__(self) -> str:
        tags: List[str] = []
        if self.suppressed:
            tags.append("SUPPRESSED")
        if self.escalated:
            tags.append("ESCALATED")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        urgency_icon = {"critical": "🔴", "normal": "🟡", "low": "🟢"}.get(self.urgency, "ℹ")
        return f"[{self.time_str}] {urgency_icon} [{self.source}] {self.message}{tag_str}"


# ---------------------------------------------------------------------------
# NotificationCenter
# ---------------------------------------------------------------------------


class NotificationCenter:
    """Central hub for all AI Helper notifications.

    Parameters
    ----------
    dedup_seconds:
        Minimum seconds between identical messages from the same source
        before the duplicate is fired again.  Default: 120 s.
    throttle_seconds:
        Minimum seconds between any two alerts from the same source+topic.
        Default: 60 s.
    escalate_count:
        Number of times an alert must fire within ``escalate_window`` seconds
        before urgency is automatically raised to ``"critical"``.
        Default: 3.
    escalate_window:
        Time window (seconds) for escalation counting.  Default: 300 s (5 min).
    history_size:
        Maximum number of notifications to keep in the in-memory history.
        Default: 500.
    on_notify:
        Optional callback invoked for every *non-suppressed* notification.
    """

    def __init__(
        self,
        dedup_seconds: float = 120.0,
        throttle_seconds: float = 60.0,
        escalate_count: int = 3,
        escalate_window: float = 300.0,
        history_size: int = 500,
        on_notify: Optional[Callable[[NotificationRecord], None]] = None,
    ) -> None:
        self.dedup_seconds = dedup_seconds
        self.throttle_seconds = throttle_seconds
        self.escalate_count = escalate_count
        self.escalate_window = escalate_window
        self.history_size = history_size
        self.on_notify = on_notify

        # In-memory history (newest at end)
        self._history: Deque[NotificationRecord] = deque(maxlen=history_size)

        # Dedup: (source, message) → last fire timestamp
        self._last_fire: Dict[Tuple[str, str], float] = {}

        # Throttle: (source, topic) → last fire timestamp
        self._last_topic: Dict[Tuple[str, str], float] = {}

        # Escalation: (source, message) → deque of fire timestamps
        self._fire_times: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def notify(
        self,
        message: str,
        source: str = "system",
        urgency: str = "normal",
        topic: str = "general",
    ) -> NotificationRecord:
        """Process and record a notification.

        The notification is suppressed if it is a recent duplicate or the
        source+topic has been throttled.  Escalation is applied when needed.
        Returns the :class:`NotificationRecord` (even if suppressed).
        """
        now = time.time()
        key = (source, message)
        topic_key = (source, topic)

        # ---- Deduplication -------------------------------------------
        last = self._last_fire.get(key, 0.0)
        if now - last < self.dedup_seconds:
            rec = NotificationRecord(
                message=message, source=source, urgency=urgency,
                topic=topic, ts=now, suppressed=True,
            )
            self._history.append(rec)
            logger.debug("Notification suppressed (dup): %s", message[:60])
            return rec

        # ---- Throttle ------------------------------------------------
        last_topic = self._last_topic.get(topic_key, 0.0)
        if now - last_topic < self.throttle_seconds:
            rec = NotificationRecord(
                message=message, source=source, urgency=urgency,
                topic=topic, ts=now, suppressed=True,
            )
            self._history.append(rec)
            logger.debug("Notification throttled: %s", message[:60])
            return rec

        # ---- Escalation ----------------------------------------------
        times = self._fire_times[key]
        times.append(now)
        recent = sum(1 for t in times if now - t <= self.escalate_window)
        escalated = False
        if recent >= self.escalate_count and urgency != "critical":
            urgency = "critical"
            escalated = True

        # ---- Record & fire -------------------------------------------
        self._last_fire[key] = now
        self._last_topic[topic_key] = now

        rec = NotificationRecord(
            message=message, source=source, urgency=urgency,
            topic=topic, ts=now, suppressed=False, escalated=escalated,
        )
        self._history.append(rec)
        logger.info("NOTIFY [%s/%s] %s", source, urgency, message[:100])

        if self.on_notify:
            try:
                self.on_notify(rec)
            except Exception:  # noqa: BLE001
                logger.exception("on_notify callback failed")

        return rec

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[NotificationRecord]:
        """All stored notifications, oldest first."""
        return list(self._history)

    def active_alerts(self) -> List[NotificationRecord]:
        """Non-suppressed alerts, newest first."""
        return [r for r in reversed(self._history) if not r.suppressed]

    def format_history(
        self,
        limit: int = 50,
        urgency: Optional[str] = None,
        source: Optional[str] = None,
        include_suppressed: bool = False,
    ) -> str:
        """Return a human-readable notification log.

        Parameters
        ----------
        limit:
            Maximum number of entries (newest first).
        urgency:
            Filter to a specific urgency level.
        source:
            Filter to a specific source.
        include_suppressed:
            Include suppressed (deduplicated/throttled) entries.
        """
        records = list(reversed(self._history))
        if not include_suppressed:
            records = [r for r in records if not r.suppressed]
        if urgency:
            records = [r for r in records if r.urgency == urgency]
        if source:
            records = [r for r in records if r.source == source]
        records = records[:limit]

        if not records:
            return "No notifications matching the filters."

        lines = [f"=== Notification History ({len(records)} shown) ==="]
        for r in records:
            lines.append(f"  {r}")
        return "\n".join(lines)

    def stats(self) -> str:
        """Return a brief statistics string."""
        total = len(self._history)
        active = sum(1 for r in self._history if not r.suppressed)
        suppressed = total - active
        critical = sum(1 for r in self._history if r.urgency == "critical" and not r.suppressed)
        escalated = sum(1 for r in self._history if r.escalated)
        return (
            f"Notifications: {active} active, {suppressed} suppressed, "
            f"{critical} critical, {escalated} escalated"
        )

    def clear_history(self) -> None:
        """Wipe the in-memory notification history."""
        self._history.clear()
        self._last_fire.clear()
        self._last_topic.clear()
        self._fire_times.clear()
