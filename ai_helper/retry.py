"""Retry and circuit-breaker utilities for AI Helper.

Two primitives are provided:

* :func:`with_retry` — a decorator that re-invokes a function on failure
  with exponential back-off and optional jitter.
* :class:`CircuitBreaker` — wraps an external service so that repeated
  failures cause the circuit to *open* (fast-fail) and automatically
  recover after a cool-down period.

Both are designed to protect AI Helper from transient network errors when
calling Ollama, Meshy, the GitHub releases API, or any other external
service.

Usage
-----
::

    from ai_helper.retry import with_retry, CircuitBreaker

    # Retry up to 3 times with exponential back-off on network errors
    @with_retry(max_attempts=3, backoff_base=1.0)
    def fetch_model_list(url: str) -> list:
        ...

    # Circuit breaker for Ollama
    ollama_cb = CircuitBreaker(name="ollama", failure_threshold=3, recovery_timeout=30)

    def safe_chat(prompt: str) -> str:
        return ollama_cb.call(lambda: _chat_impl(prompt))
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    jitter: bool = True,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[[F], F]:
    """Decorator: retry *func* up to *max_attempts* times on *exceptions*.

    Back-off
    --------
    After attempt *n* the decorator sleeps for::

        min(backoff_base * 2^(n-1), backoff_max)  [+ optional jitter]

    Parameters
    ----------
    max_attempts:
        Maximum number of calls (including the first).  ``1`` means no
        retries.
    backoff_base:
        Initial sleep in seconds (doubled on each subsequent failure).
    backoff_max:
        Cap on the sleep duration in seconds.
    jitter:
        When ``True``, randomise the sleep by ±25 % to prevent
        thundering-herd after simultaneous failures.
    exceptions:
        Tuple of exception types that trigger a retry.  Defaults to
        ``(Exception,)`` which catches everything except ``BaseException``
        sub-classes like ``SystemExit`` and ``KeyboardInterrupt``.
    on_retry:
        Optional callback ``on_retry(attempt_number, exc)`` invoked before
        each retry sleep.  Useful for logging or metrics.

    Example
    -------
    ::

        @with_retry(max_attempts=4, backoff_base=0.5, exceptions=(OSError,))
        def read_remote_file(url: str) -> bytes:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    sleep = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                    if jitter:
                        sleep *= 0.75 + random.random() * 0.5  # ±25%
                    if on_retry:
                        on_retry(attempt, exc)
                    logger.debug(
                        "Retry %d/%d for %s after %.1fs (error: %s)",
                        attempt, max_attempts, func.__qualname__, sleep, exc,
                    )
                    time.sleep(sleep)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class _State(Enum):
    CLOSED = auto()    # normal operation
    OPEN = auto()      # failing fast
    HALF_OPEN = auto() # testing recovery


class CircuitBreaker:
    """Wrap external service calls to fail fast when the service is down.

    State machine::

        CLOSED → (failure_threshold failures) → OPEN
        OPEN   → (recovery_timeout elapsed)   → HALF_OPEN
        HALF_OPEN → (success)  → CLOSED
        HALF_OPEN → (failure)  → OPEN

    Parameters
    ----------
    name:
        Human-readable name of the service (used in log messages).
    failure_threshold:
        How many consecutive failures before the circuit opens.
    recovery_timeout:
        Seconds to wait in OPEN state before attempting a probe call.
    success_threshold:
        How many consecutive successes in HALF_OPEN before fully closing.

    Example
    -------
    ::

        cb = CircuitBreaker("ollama", failure_threshold=3, recovery_timeout=30)

        def safe_generate(prompt):
            return cb.call(lambda: client.generate(prompt))
    """

    def __init__(
        self,
        name: str = "service",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = _State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Return the current state as a string: ``"CLOSED"``, ``"OPEN"``, or ``"HALF_OPEN"``."""
        return self._state.name

    @property
    def is_open(self) -> bool:
        """Return ``True`` if the circuit is open (failing fast)."""
        return self._state == _State.OPEN

    def call(self, func: Callable[[], Any]) -> Any:
        """Call *func* through the circuit breaker.

        Raises :exc:`CircuitOpenError` immediately when the circuit is open
        and the recovery timeout has not yet elapsed.

        Parameters
        ----------
        func:
            Zero-argument callable to protect.

        Returns
        -------
        Any
            The return value of *func* on success.

        Raises
        ------
        CircuitOpenError
            When the circuit is open and not yet ready for a probe call.
        Any exception raised by *func*
            Propagated after the failure counter is updated.
        """
        with self._lock:
            state = self._check_state()

        if state == _State.OPEN:
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN — service appears to be down. "
                f"Will retry after {self.recovery_timeout:.0f}s cool-down."
            )

        try:
            result = func()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc from None

    def reset(self) -> None:
        """Manually close the circuit (clear failure count)."""
        with self._lock:
            self._state = _State.CLOSED
            self._failure_count = 0
            self._success_count = 0
        logger.info("Circuit '%s' manually reset to CLOSED.", self.name)

    def __str__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self.state}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_state(self) -> _State:
        if self._state == _State.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = _State.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit '%s' → HALF_OPEN (probe call allowed).", self.name)
        return self._state

    def _on_success(self) -> None:
        with self._lock:
            if self._state == _State.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = _State.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit '%s' → CLOSED (service recovered).", self.name)
            elif self._state == _State.CLOSED:
                self._failure_count = 0  # reset on any success

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state in (_State.CLOSED, _State.HALF_OPEN):
                if self._failure_count >= self.failure_threshold:
                    self._state = _State.OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit '%s' → OPEN after %d failures. "
                        "Fast-failing for %.0fs.",
                        self.name, self._failure_count, self.recovery_timeout,
                    )


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit breaker is open."""
