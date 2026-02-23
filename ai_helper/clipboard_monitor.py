"""Clipboard monitor.

Watches the system clipboard for changes and intelligently responds:

* **File paths** — offers to open or read the file via the agent.
* **Shell commands** — offers to execute them.
* **Python tracebacks / error messages** — offers to explain or fix with Ollama.
* **URLs** — offers to summarise or fetch.
* **General text** — makes the text available to the agent as context.

Works on all platforms without extra packages as a first option:

* **Windows** — ``ctypes`` / ``win32clipboard`` via ``ctypes``
* **macOS** — ``pbpaste`` subprocess
* **Linux** — ``xclip`` / ``xsel`` / ``wl-paste`` subprocess

If ``pyperclip`` is installed it is used as a higher-quality fallback.

Usage
-----
::

    from ai_helper.clipboard_monitor import ClipboardMonitor

    def handle(event):
        print(f"Clipboard changed: {event.kind} — {event.text[:60]}")

    mon = ClipboardMonitor(callback=handle)
    mon.start()
    # … runs in background …
    mon.stop()
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ClipboardEvent:
    """A clipboard change event."""
    text: str
    kind: str   # "file_path" | "command" | "error" | "url" | "text"
    ts: float = field(default_factory=time.time)

    @property
    def time_str(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))

    def __str__(self) -> str:
        preview = self.text[:80].replace("\n", "↵")
        return f"[{self.time_str}] clipboard:{self.kind}  {preview!r}"


# ---------------------------------------------------------------------------
# Platform clipboard readers
# ---------------------------------------------------------------------------


def _read_clipboard_pyperclip() -> Optional[str]:
    try:
        import pyperclip  # type: ignore[import-untyped]
        return pyperclip.paste()
    except Exception:  # noqa: BLE001
        return None


def _read_clipboard_windows() -> Optional[str]:
    import ctypes  # noqa: PLC0415
    CF_UNICODETEXT = 13
    try:
        ctypes.windll.user32.OpenClipboard(0)  # type: ignore[attr-defined]
        handle = ctypes.windll.user32.GetClipboardData(CF_UNICODETEXT)  # type: ignore[attr-defined]
        if not handle:
            return None
        pdata = ctypes.c_wchar_p(handle)
        return pdata.value
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            ctypes.windll.user32.CloseClipboard()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _read_clipboard_macos() -> Optional[str]:
    try:
        return subprocess.check_output(["pbpaste"], text=True, timeout=2)  # noqa: S603
    except Exception:  # noqa: BLE001
        return None


def _read_clipboard_linux() -> Optional[str]:
    for cmd in (["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"],
                ["wl-paste", "--no-newline"]):
        try:
            return subprocess.check_output(cmd, text=True, timeout=2,  # noqa: S603
                                           stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            continue
    return None


def read_clipboard() -> Optional[str]:
    """Read current clipboard text, trying pyperclip first then platform APIs."""
    text = _read_clipboard_pyperclip()
    if text is not None:
        return text
    if _SYSTEM == "Windows":
        return _read_clipboard_windows()
    if _SYSTEM == "Darwin":
        return _read_clipboard_macos()
    return _read_clipboard_linux()


# ---------------------------------------------------------------------------
# Content classifier
# ---------------------------------------------------------------------------


_FILE_PATH_RE = re.compile(
    r"^([A-Za-z]:[\\\/][^\n\r\"'<>|?*]{2,}|"
    r"\/[a-zA-Z0-9_.~\-\/]{2,}|"
    r"~\/[^\n\r\"'<>|?*]{2,})$"
)
_URL_RE = re.compile(r"^https?://\S+$")
_CMD_RE = re.compile(r"^(pip|python|python3|git|npm|node|docker|kubectl|curl|wget|"
                     r"bash|sh|pwsh|powershell|cd |ls |dir |rm |cp |mv |echo |cat )")
_ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"Error:|Exception:|SyntaxError:|TypeError:|ValueError:|"
    r"FileNotFoundError:|ImportError:|FAILED|fatal:)",
    re.IGNORECASE,
)


def classify(text: str) -> str:
    """Return a content-kind label for clipboard text."""
    stripped = text.strip()
    if _FILE_PATH_RE.match(stripped):
        return "file_path"
    if _URL_RE.match(stripped):
        return "url"
    if _CMD_RE.match(stripped):
        return "command"
    if _ERROR_RE.search(stripped):
        return "error"
    return "text"


# ---------------------------------------------------------------------------
# ClipboardMonitor
# ---------------------------------------------------------------------------


class ClipboardMonitor:
    """Background thread that polls the clipboard and fires a callback on changes.

    Parameters
    ----------
    callback:
        Function called with each :class:`ClipboardEvent`.
    poll_interval:
        Seconds between clipboard reads (default 1.5 s).
    min_length:
        Ignore clipboard content shorter than this (default 3 chars).
    """

    def __init__(
        self,
        callback: Callable[[ClipboardEvent], None],
        poll_interval: float = 1.5,
        min_length: int = 3,
    ) -> None:
        self.callback = callback
        self.poll_interval = poll_interval
        self.min_length = min_length

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_text: Optional[str] = None

    def start(self) -> None:
        """Start monitoring the clipboard in a background daemon thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ai-helper-clipboard", daemon=True
        )
        self._thread.start()
        logger.info("Clipboard monitor started (interval=%.1fs)", self.poll_interval)

    def stop(self) -> None:
        """Stop the clipboard monitor."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 2)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = read_clipboard()
                if text and len(text) >= self.min_length and text != self._last_text:
                    self._last_text = text
                    kind = classify(text)
                    event = ClipboardEvent(text=text, kind=kind)
                    logger.debug("Clipboard event: %s", event)
                    try:
                        self.callback(event)
                    except Exception:  # noqa: BLE001
                        logger.exception("Clipboard callback error")
            except Exception:  # noqa: BLE001
                logger.debug("Clipboard read error", exc_info=True)
            self._stop.wait(timeout=self.poll_interval)
