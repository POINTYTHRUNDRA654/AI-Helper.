"""Inter-process communicator.

Provides two complementary communication mechanisms:

1. **Message bus** – an in-process publish/subscribe bus so that the
   modules inside AI Helper can exchange typed messages without tight
   coupling.

2. **Desktop notifications** – a thin wrapper that emits a desktop
   notification (via ``notify-send`` on Linux, ``osascript`` on macOS, or
   ``msg`` on Windows) so the user can see alerts without a GUI window.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message / bus
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """An immutable message passed between AI Helper components."""

    topic: str
    payload: Any
    source: str = "ai_helper"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.source} → {self.topic}: {self.payload}"


class MessageBus:
    """Simple synchronous publish/subscribe message bus.

    Usage::

        bus = MessageBus()
        bus.subscribe("alert", lambda msg: print(msg))
        bus.publish(Message(topic="alert", payload="CPU high!"))
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = {}
        self._history: List[Message] = []

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Register *handler* to be called when a message on *topic* is published."""
        self._subscribers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Remove a previously registered *handler* for *topic*."""
        handlers = self._subscribers.get(topic, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def publish(self, message: Message) -> None:
        """Publish *message* to all subscribers of its topic.

        Also publishes to the wildcard ``"*"`` topic so callers can
        listen to every message with a single subscription.
        """
        self._history.append(message)
        for handler in list(self._subscribers.get(message.topic, [])):
            try:
                handler(message)
            except Exception:  # noqa: BLE001
                logger.exception("Error in message handler for topic %r", message.topic)
        for handler in list(self._subscribers.get("*", [])):
            try:
                handler(message)
            except Exception:  # noqa: BLE001
                logger.exception("Error in wildcard message handler")

    def history(self, topic: Optional[str] = None, limit: int = 100) -> List[Message]:
        """Return recent messages, optionally filtered by *topic*."""
        msgs = self._history if topic is None else [m for m in self._history if m.topic == topic]
        return msgs[-limit:]


# ---------------------------------------------------------------------------
# Desktop notifications
# ---------------------------------------------------------------------------


class Notifier:
    """Send desktop notifications using the OS native notification system.

    Falls back to a log message when no supported notifier is available so
    the application never crashes in a headless environment.
    """

    def __init__(self, app_name: str = "AI Helper") -> None:
        self.app_name = app_name
        self._system = platform.system()

    def notify(self, title: str, message: str, urgency: str = "normal") -> bool:
        """Send a desktop notification.

        Parameters
        ----------
        title:
            Short heading shown in bold.
        message:
            Body text of the notification.
        urgency:
            ``"low"``, ``"normal"`` or ``"critical"`` (Linux only).

        Returns
        -------
        bool
            ``True`` if a notification was dispatched, ``False`` if we
            fell back to logging.
        """
        try:
            if self._system == "Linux" and shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", "--app-name", self.app_name, f"--urgency={urgency}", title, message],
                    check=False,
                    timeout=5,
                )
                return True
            if self._system == "Darwin":
                script = (
                    f'display notification "{message}" '
                    f'with title "{title}" '
                    f'subtitle "{self.app_name}"'
                )
                subprocess.run(["osascript", "-e", script], check=False, timeout=5)
                return True
            if self._system == "Windows":
                # msg * requires local session; best-effort only
                subprocess.run(
                    ["msg", "*", f"{title}: {message}"], check=False, timeout=5
                )
                return True
        except Exception:  # noqa: BLE001
            pass

        # Headless / unsupported – fall back to logging
        log_level = logging.WARNING if urgency == "critical" else logging.INFO
        logger.log(log_level, "[Notification] %s – %s", title, message)
        return False


# ---------------------------------------------------------------------------
# Convenience façade
# ---------------------------------------------------------------------------


class Communicator:
    """Combines the :class:`MessageBus` and :class:`Notifier` into one object.

    The orchestrator and other modules interact with this single object to
    both route internal messages and surface alerts to the user.
    """

    def __init__(self, app_name: str = "AI Helper") -> None:
        self.bus = MessageBus()
        self.notifier = Notifier(app_name=app_name)

    def alert(
        self,
        message: str,
        source: str = "ai_helper",
        urgency: str = "normal",
        topic: str = "alert",
    ) -> None:
        """Publish an alert on the bus **and** send a desktop notification."""
        self.bus.publish(Message(topic=topic, payload=message, source=source))
        self.notifier.notify(title=source, message=message, urgency=urgency)

    def publish(self, topic: str, payload: Any, source: str = "ai_helper") -> None:
        """Publish an arbitrary message on the internal bus."""
        self.bus.publish(Message(topic=topic, payload=payload, source=source))

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to internal bus messages."""
        self.bus.subscribe(topic, handler)
