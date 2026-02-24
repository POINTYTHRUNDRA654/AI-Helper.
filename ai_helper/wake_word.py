"""Wake-word listener (prototype).

Listens on the default microphone for a wake word (e.g. "mossy").
When heard, it captures the next utterance as a command and invokes a
callback. Uses `speech_recognition` if available; otherwise it is a no-op.

This is a best-effort prototype and depends on microphone quality and
network connectivity (Google Speech API). It is resilient to failures and
continues listening after errors.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency
    import speech_recognition as sr  # type: ignore
except Exception:  # noqa: BLE001
    sr = None

try:  # Optional dependency for local STT
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # noqa: BLE001
    WhisperModel = None


class WakeWordListener:
    """Background listener for a wake word followed by a spoken command."""

    def __init__(
        self,
        wake_word: str = "mossy",
        phrase_time_limit: float = 6.0,
        device_index: int | None = None,
        backend: str = "google",  # "google" | "whisper"
        whisper_model: str = "small.en",
    ) -> None:
        self.wake_word = wake_word.lower()
        self.phrase_time_limit = phrase_time_limit
        self.device_index = device_index
        self.backend = backend
        self.whisper_model_name = whisper_model
        self._whisper_model: WhisperModel | None = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        """Return True if speech_recognition is installed."""
        if self.backend == "google":
            return sr is not None
        if self.backend == "whisper":
            return sr is not None and WhisperModel is not None
        return False

    def start(self, on_command: Callable[[str], None]) -> None:
        """Start the listener in a daemon thread."""
        if sr is None:
            logger.warning("Wake-word listener unavailable: install SpeechRecognition + PyAudio")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, args=(on_command,), daemon=True, name="ai-helper-wake-word"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait briefly."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self, on_command: Callable[[str], None]) -> None:
        assert sr is not None  # for type checkers
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone(device_index=self.device_index) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info(
                    "Wake-word listener active (wake word: %s, backend: %s, device: %s)",
                    self.wake_word,
                    self.backend,
                    self.device_index if self.device_index is not None else "default",
                )
                while not self._stop.is_set():
                    try:
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=self.phrase_time_limit)
                        text = self._recognize(audio, recognizer)
                        if not text:
                            continue
                        lowered = text.lower()
                        if self.wake_word in lowered:
                            logger.info("Wake word detected: %s", text)
                            # Capture the next phrase as the command.
                            cmd_audio = recognizer.listen(
                                source,
                                timeout=5,
                                phrase_time_limit=self.phrase_time_limit,
                            )
                            cmd_text = self._recognize(cmd_audio, recognizer)
                            if cmd_text:
                                threading.Thread(
                                    target=on_command, args=(cmd_text,), daemon=True, name="ai-helper-wake-command"
                                ).start()
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Wake-word loop error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wake-word listener failed to start: %s", exc)

    def _recognize(self, audio: "sr.AudioData", recognizer: "sr.Recognizer") -> str:
        if self.backend == "google":
            return self._recognize_google(audio, recognizer)
        if self.backend == "whisper":
            return self._recognize_whisper(audio)
        return ""

    def _recognize_google(self, audio: "sr.AudioData", recognizer: "sr.Recognizer") -> str:
        try:
            return recognizer.recognize_google(audio)
        except Exception:  # noqa: BLE001
            return ""

    def _recognize_whisper(self, audio: "sr.AudioData") -> str:
        if WhisperModel is None:
            return ""
        try:
            if self._whisper_model is None:
                self._whisper_model = WhisperModel(self.whisper_model_name, device="cpu", compute_type="int8")
            # Convert audio bytes to numpy int16 then float32 in range [-1, 1]
            raw = audio.get_raw_data()
            np_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = audio.sample_rate
            segments, _ = self._whisper_model.transcribe(np_audio, language="en", beam_size=1, vad_filter=True, word_timestamps=False, temperature=0.0, condition_on_previous_text=False, initial_prompt=None, without_timestamps=True, vad_parameters={"min_silence_duration_ms": 500}, speed_up=False, sample_rate=sample_rate)
            text = " ".join(seg.text.strip() for seg in segments) if segments else ""
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Whisper recognize failed: %s", exc)
            return ""
