"""Global hotkey manager.

Registers system-wide keyboard shortcuts that trigger AI Helper actions
from anywhere on the desktop — even when AI Helper's terminal is not in focus.

Requires the optional ``pynput`` package::

    pip install pynput

If ``pynput`` is not installed, :class:`HotkeyManager` still imports and
constructs successfully but ``start()`` logs a warning and returns without
registering any hotkeys.

Default hotkeys
---------------
+----------------------------+----------------------------------------------+
| Shortcut                   | Action                                       |
+============================+==============================================+
| ``Ctrl+Alt+A``             | Ask the agent a question (prompts via stdin) |
| ``Ctrl+Alt+S``             | Speak current system status aloud            |
| ``Ctrl+Alt+O``             | Organise the desktop right now               |
| ``Ctrl+Alt+N``             | Read the latest alert aloud                  |
| ``Ctrl+Alt+G``             | Speak current GPU stats                      |
+----------------------------+----------------------------------------------+

All defaults can be overridden via ``HotkeyManager(hotkeys={...})``.

Usage
-----
::

    from ai_helper.hotkey import HotkeyManager

    mgr = HotkeyManager(
        on_ask=lambda: agent.execute(input("Ask: ")),
        on_status=lambda: print(monitor.format_snapshot()),
    )
    mgr.start()
    # … runs in background thread …
    mgr.stop()
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key combo constants  (used when building pynput GlobalHotKeys dict)
# ---------------------------------------------------------------------------

HOTKEY_ASK     = "<ctrl>+<alt>+a"
HOTKEY_STATUS  = "<ctrl>+<alt>+s"
HOTKEY_ORGANISE = "<ctrl>+<alt>+o"
HOTKEY_NOTIFY  = "<ctrl>+<alt>+n"
HOTKEY_GPU     = "<ctrl>+<alt>+g"


def _noop() -> None:
    """Default no-op action."""


class HotkeyManager:
    """Register global keyboard shortcuts and bind them to AI Helper actions.

    Parameters
    ----------
    on_ask:
        Called when the user presses the 'ask' hotkey.
    on_status:
        Called when the user presses the 'status' hotkey.
    on_organise:
        Called when the user presses the 'organise' hotkey.
    on_notify:
        Called when the user presses the 'latest notification' hotkey.
    on_gpu:
        Called when the user presses the 'GPU stats' hotkey.
    hotkeys:
        Optional dict of ``{pynput_combo: callable}`` to use instead of the
        defaults.  Overrides the individual ``on_*`` parameters.
    """

    def __init__(
        self,
        on_ask: Callable[[], None] = _noop,
        on_status: Callable[[], None] = _noop,
        on_organise: Callable[[], None] = _noop,
        on_notify: Callable[[], None] = _noop,
        on_gpu: Callable[[], None] = _noop,
        hotkeys: Optional[Dict[str, Callable[[], None]]] = None,
    ) -> None:
        self._bindings: Dict[str, Callable[[], None]] = hotkeys or {
            HOTKEY_ASK:      on_ask,
            HOTKEY_STATUS:   on_status,
            HOTKEY_ORGANISE: on_organise,
            HOTKEY_NOTIFY:   on_notify,
            HOTKEY_GPU:      on_gpu,
        }
        self._listener = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start listening for global hotkeys in a background thread.

        Returns ``True`` if hotkeys were registered, ``False`` if pynput
        is unavailable.
        """
        try:
            from pynput import keyboard  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "pynput is not installed — global hotkeys disabled. "
                "Install with: pip install pynput"
            )
            return False

        # Wrap each callback in an exception-safe wrapper running in its own thread
        safe_bindings: Dict[str, Callable[[], None]] = {}
        for combo, action in self._bindings.items():
            def _make_handler(fn: Callable[[], None]) -> Callable[[], None]:
                def handler() -> None:
                    threading.Thread(
                        target=self._safe_call, args=(fn,),
                        daemon=True
                    ).start()
                return handler
            safe_bindings[combo] = _make_handler(action)

        self._listener = keyboard.GlobalHotKeys(safe_bindings)
        self._thread = threading.Thread(
            target=self._listener.run, name="ai-helper-hotkeys", daemon=True
        )
        self._thread.start()
        logger.info(
            "Global hotkeys registered: %s", ", ".join(self._bindings.keys())
        )
        return True

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Hotkey manager stopped")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def bindings_info(self) -> str:
        """Return a human-readable list of registered hotkeys."""
        lines = ["=== Global Hotkeys ==="]
        descriptions = {
            HOTKEY_ASK:      "Ask the AI agent a question",
            HOTKEY_STATUS:   "Speak current system status",
            HOTKEY_ORGANISE: "Organise the desktop now",
            HOTKEY_NOTIFY:   "Read latest notification",
            HOTKEY_GPU:      "Speak GPU statistics",
        }
        for combo in self._bindings:
            desc = descriptions.get(combo, "custom action")
            lines.append(f"  {combo:25s}  {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_call(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            logger.exception("Hotkey action raised an exception")
