"""Text-to-speech voice module.

Gives AI Helper a voice so it can read alerts, status summaries and
notifications out loud — keeping the user informed without them having
to watch a screen.

Engine priority
---------------
1. **pyttsx3** – offline, cross-platform (SAPI 5 on Windows,
   NSSpeechSynthesizer on macOS, eSpeak on Linux).  Best quality.
2. **OS command-line tools** – ``say`` (macOS), ``espeak`` (Linux),
   ``PowerShell Add-Type`` (Windows) — used when pyttsx3 is not installed.
3. **Log fallback** – if no TTS engine is available the text is simply
   logged at INFO level so the app never crashes in a headless environment.

The :class:`Speaker` runs speech in a background daemon thread via a
``queue.Queue`` so calls to :meth:`speak` are fully non-blocking and the
monitoring loop is never stalled by a slow TTS engine.
"""

from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()

# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------


def _try_import_pyttsx3():
    """Return the pyttsx3 module, or None if not installed."""
    try:
        import pyttsx3  # type: ignore[import-untyped]
        return pyttsx3
    except Exception:  # noqa: BLE001
        return None


def _cli_fallback_available() -> bool:
    """Return True if at least one command-line TTS tool is on PATH."""
    if _SYSTEM == "Darwin" and shutil.which("say"):
        return True
    if _SYSTEM == "Linux" and shutil.which("espeak"):
        return True
    if _SYSTEM == "Windows":
        return True  # PowerShell is always available on Windows
    return False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class VoiceSettings:
    """Configurable TTS parameters."""

    rate: int = 175          # words per minute
    volume: float = 1.0      # 0.0 – 1.0
    voice_id: Optional[str] = None  # pyttsx3 voice id; None = engine default


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------


class Speaker:
    """Non-blocking text-to-speech speaker.

    Enqueue text with :meth:`speak` (returns immediately) or call
    :meth:`speak_now` to block until the utterance finishes.

    Parameters
    ----------
    settings:
        Optional :class:`VoiceSettings` to configure rate, volume and
        voice.  Defaults are used when not provided.
    enabled:
        Set to ``False`` to silence all speech without removing the
        object from the communicator pipeline (e.g. ``--no-voice`` flag).
    """

    # Sentinel pushed into the queue to stop the worker thread.
    _STOP = object()

    def __init__(
        self,
        settings: Optional[VoiceSettings] = None,
        enabled: bool = True,
    ) -> None:
        self.settings = settings or VoiceSettings()
        self.enabled = enabled

        self._queue: queue.Queue[object] = queue.Queue()
        self._pyttsx3 = _try_import_pyttsx3()
        self._engine = None  # initialised lazily inside the worker thread
        self._lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run, name="ai-helper-speaker", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Enqueue *text* for speech.  Returns immediately."""
        if self.enabled and text:
            self._queue.put(text)

    def speak_now(self, text: str, timeout: float = 30.0) -> bool:
        """Speak *text* and block until it finishes (or *timeout* seconds pass).

        Returns ``True`` if speech completed, ``False`` on timeout or if
        the speaker is disabled.
        """
        if not self.enabled or not text:
            return False
        done = threading.Event()

        def _say() -> None:
            self._utter(text)
            done.set()

        t = threading.Thread(target=_say, daemon=True)
        t.start()
        return done.wait(timeout=timeout)

    def stop(self) -> None:
        """Clear the speech queue and stop any in-progress utterance."""
        with self._queue.mutex:
            self._queue.queue.clear()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        """Signal the worker thread to exit and wait for it."""
        self._queue.put(self._STOP)
        self._worker.join(timeout=5)

    def list_voices(self) -> list[str]:
        """Return the names of all available pyttsx3 voices.

        Returns an empty list when pyttsx3 is not installed.
        """
        if self._pyttsx3 is None:
            return []
        try:
            engine = self._pyttsx3.init()
            voices = [v.name for v in engine.getProperty("voices")]
            engine.stop()
            return voices
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread: dequeue and utter text one item at a time."""
        # Initialise pyttsx3 engine inside this thread (required by pyttsx3).
        if self._pyttsx3 is not None:
            try:
                self._engine = self._pyttsx3.init()
                self._apply_settings(self._engine)
            except Exception:  # noqa: BLE001
                logger.warning("pyttsx3 engine init failed; will use CLI fallback")
                self._engine = None

        while True:
            item = self._queue.get()
            if item is self._STOP:
                break
            try:
                self._utter(str(item))
            except Exception:  # noqa: BLE001
                logger.exception("Speaker error while uttering text")
            finally:
                self._queue.task_done()

    def _utter(self, text: str) -> None:
        """Speak *text* using the best available engine (called from any thread)."""
        if self._engine is not None:
            self._utter_pyttsx3(text)
        elif _cli_fallback_available():
            self._utter_cli(text)
        else:
            logger.info("[TTS] %s", text)

    def _utter_pyttsx3(self, text: str) -> None:
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:  # noqa: BLE001
                logger.warning("pyttsx3 utter failed; retrying via CLI")
                self._utter_cli(text)

    def _utter_cli(self, text: str) -> None:
        """Speak using a platform command-line TTS tool."""
        safe = text.replace('"', "'")  # basic quote sanitisation for shell args
        try:
            if _SYSTEM == "Darwin" and shutil.which("say"):
                rate_arg = str(int(self.settings.rate))
                subprocess.run(
                    ["say", "-r", rate_arg, safe],
                    check=False, timeout=60,
                )
            elif _SYSTEM == "Linux" and shutil.which("espeak"):
                rate_arg = str(int(self.settings.rate))
                vol_arg = str(int(self.settings.volume * 100))
                subprocess.run(
                    ["espeak", "-s", rate_arg, "-a", vol_arg, safe],
                    check=False, timeout=60,
                )
            elif _SYSTEM == "Windows":
                ps_script = (
                    "Add-Type -AssemblyName System.speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Rate = {self._ps_rate()}; "
                    f"$s.Volume = {int(self.settings.volume * 100)}; "
                    f'$s.Speak("{safe}");'
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    check=False, timeout=60,
                )
            else:
                logger.info("[TTS] %s", text)
        except subprocess.TimeoutExpired:
            logger.warning("TTS CLI timed out for text: %r", text[:60])
        except Exception:  # noqa: BLE001
            logger.exception("TTS CLI error")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_settings(self, engine) -> None:  # noqa: ANN001
        try:
            engine.setProperty("rate", self.settings.rate)
            engine.setProperty("volume", self.settings.volume)
            if self.settings.voice_id:
                engine.setProperty("voice", self.settings.voice_id)
        except Exception:  # noqa: BLE001
            logger.debug("Could not apply all voice settings")

    def _ps_rate(self) -> int:
        """Map words-per-minute to PowerShell SpeechSynthesizer Rate (-10..10)."""
        # pyttsx3 default is 200 wpm ≈ Rate 0.  Clamp to [-10, 10].
        rate = int((self.settings.rate - 200) / 20)
        return max(-10, min(10, rate))
