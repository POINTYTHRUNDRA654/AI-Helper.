"""Tests for ai_helper.retry (CircuitBreaker, with_retry)."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from ai_helper.retry import CircuitBreaker, CircuitOpenError, with_retry


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------


class TestWithRetry(unittest.TestCase):
    def test_succeeds_first_attempt(self):
        calls = []
        @with_retry(max_attempts=3, backoff_base=0.0)
        def func():
            calls.append(1)
            return "ok"
        result = func()
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_on_exception_then_succeeds(self):
        calls = []
        @with_retry(max_attempts=3, backoff_base=0.0, jitter=False)
        def func():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("transient")
            return "recovered"
        result = func()
        self.assertEqual(result, "recovered")
        self.assertEqual(len(calls), 3)

    def test_raises_after_max_attempts(self):
        calls = []
        @with_retry(max_attempts=2, backoff_base=0.0, jitter=False)
        def func():
            calls.append(1)
            raise ValueError("always fails")
        with self.assertRaises(ValueError):
            func()
        self.assertEqual(len(calls), 2)

    def test_only_retries_specified_exceptions(self):
        @with_retry(max_attempts=3, backoff_base=0.0, exceptions=(OSError,))
        def func():
            raise TypeError("wrong type")
        with self.assertRaises(TypeError):
            func()

    def test_on_retry_callback(self):
        retries = []
        @with_retry(max_attempts=3, backoff_base=0.0, jitter=False,
                    on_retry=lambda n, e: retries.append((n, str(e))))
        def func():
            raise OSError("oops")
        with self.assertRaises(OSError):
            func()
        self.assertEqual(len(retries), 2)  # called before each retry, not on final fail
        self.assertEqual(retries[0][0], 1)
        self.assertEqual(retries[1][0], 2)

    def test_max_attempts_one_no_retry(self):
        calls = []
        @with_retry(max_attempts=1, backoff_base=0.0)
        def func():
            calls.append(1)
            raise RuntimeError("fail")
        with self.assertRaises(RuntimeError):
            func()
        self.assertEqual(len(calls), 1)

    def test_preserves_return_value(self):
        @with_retry(max_attempts=2, backoff_base=0.0)
        def func():
            return {"key": "value"}
        self.assertEqual(func(), {"key": "value"})

    def test_preserves_function_name(self):
        @with_retry()
        def my_special_function():
            pass
        self.assertEqual(my_special_function.__name__, "my_special_function")


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker(unittest.TestCase):
    def _breaker(self, threshold=3, recovery=0.05):
        return CircuitBreaker("test", failure_threshold=threshold,
                               recovery_timeout=recovery)

    def test_starts_closed(self):
        cb = self._breaker()
        self.assertEqual(cb.state, "CLOSED")
        self.assertFalse(cb.is_open)

    def test_successful_calls_stay_closed(self):
        cb = self._breaker()
        for _ in range(10):
            result = cb.call(lambda: "ok")
            self.assertEqual(result, "ok")
        self.assertEqual(cb.state, "CLOSED")

    def test_opens_after_failure_threshold(self):
        cb = self._breaker(threshold=3)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        self.assertEqual(cb.state, "OPEN")
        self.assertTrue(cb.is_open)

    def test_open_circuit_raises_circuit_open_error(self):
        cb = self._breaker(threshold=1)
        with self.assertRaises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        with self.assertRaises(CircuitOpenError):
            cb.call(lambda: "should not run")

    def test_transitions_to_half_open_after_timeout(self):
        cb = self._breaker(threshold=1, recovery=0.05)
        with self.assertRaises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        self.assertEqual(cb.state, "OPEN")
        time.sleep(0.08)
        # Force state check by calling (should probe)
        cb.call(lambda: None)  # probe call succeeds
        self.assertEqual(cb.state, "CLOSED")

    def test_closes_after_successful_probe(self):
        cb = self._breaker(threshold=1, recovery=0.05)
        with self.assertRaises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        time.sleep(0.08)
        cb.call(lambda: "probe success")
        self.assertEqual(cb.state, "CLOSED")
        self.assertFalse(cb.is_open)

    def test_reopens_on_failure_in_half_open(self):
        cb = self._breaker(threshold=1, recovery=0.05)
        with self.assertRaises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        time.sleep(0.08)
        # Probe fails → should re-open
        with self.assertRaises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("still failing")))
        self.assertEqual(cb.state, "OPEN")

    def test_reset_clears_failure_count(self):
        cb = self._breaker(threshold=2)
        with self.assertRaises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        self.assertEqual(cb._failure_count, 1)
        cb.reset()
        self.assertEqual(cb.state, "CLOSED")
        self.assertEqual(cb._failure_count, 0)

    def test_str_representation(self):
        cb = self._breaker()
        text = str(cb)
        self.assertIn("test", text)
        self.assertIn("CLOSED", text)

    def test_thread_safe(self):
        """Multiple threads calling simultaneously should not corrupt state."""
        cb = self._breaker(threshold=100)
        results = []
        errors = []

        def worker():
            try:
                r = cb.call(lambda: "ok")
                results.append(r)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 20)
        self.assertEqual(errors, [])
        self.assertEqual(cb.state, "CLOSED")


# ---------------------------------------------------------------------------
# Integration: with_retry + CircuitBreaker
# ---------------------------------------------------------------------------


class TestRetryWithCircuitBreaker(unittest.TestCase):
    def test_retry_inside_circuit_breaker(self):
        """Circuit breaker wrapping a retrying function."""
        cb = CircuitBreaker("combo", failure_threshold=10)
        call_count = [0]

        @with_retry(max_attempts=3, backoff_base=0.0, jitter=False)
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise OSError("transient")
            return "done"

        result = cb.call(flaky)
        self.assertEqual(result, "done")
        self.assertEqual(call_count[0], 3)
        self.assertEqual(cb.state, "CLOSED")


if __name__ == "__main__":
    unittest.main()
