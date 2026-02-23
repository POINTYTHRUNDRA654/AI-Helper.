"""Machine-learning engine.

Provides three capabilities without any external ML libraries:

1. **Anomaly detection** – a z-score detector backed by an Exponentially
   Weighted Moving Average (EWMA) so it adapts to the host's normal load.

2. **Trend prediction** – ordinary-least-squares linear regression over
   the rolling observation window to predict whether a metric will breach
   a threshold in the next N seconds.

3. **Problem solver** – maps detected anomalies / predictions to ranked,
   actionable recommendations using a rule-weight system that is updated
   from feedback so it *learns* over time which suggestions are most useful.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Rolling statistics helpers
# ---------------------------------------------------------------------------


class RollingWindow:
    """Fixed-capacity deque that also tracks an EWMA and variance online.

    Parameters
    ----------
    maxlen:
        Maximum number of samples to keep.
    alpha:
        EWMA smoothing factor in (0, 1].  Closer to 1 = less smoothing.
    """

    def __init__(self, maxlen: int = 60, alpha: float = 0.1) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)
        self.alpha = alpha
        self._ewma: Optional[float] = None
        self._ewma_var: float = 0.0

    def push(self, value: float) -> None:
        self._buf.append(value)
        if self._ewma is None:
            self._ewma = value
            self._ewma_var = 0.0
        else:
            diff = value - self._ewma
            self._ewma += self.alpha * diff
            self._ewma_var = (1 - self.alpha) * (self._ewma_var + self.alpha * diff * diff)

    @property
    def values(self) -> List[float]:
        return list(self._buf)

    @property
    def ewma(self) -> Optional[float]:
        return self._ewma

    @property
    def ewma_std(self) -> float:
        return math.sqrt(self._ewma_var) if self._ewma_var >= 0 else 0.0

    def mean(self) -> float:
        return statistics.mean(self._buf) if self._buf else 0.0

    def stdev(self) -> float:
        return statistics.pstdev(self._buf) if len(self._buf) > 1 else 0.0

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


@dataclass
class Anomaly:
    metric: str
    value: float
    z_score: float
    ewma: float
    ewma_std: float
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"Anomaly in {self.metric!r}: value={self.value:.1f}  "
            f"z={self.z_score:.2f}  ewma={self.ewma:.1f}±{self.ewma_std:.1f}"
        )


class AnomalyDetector:
    """Per-metric EWMA z-score anomaly detector.

    A data point is anomalous when its z-score (distance from the EWMA
    expressed in units of EWMA-std) exceeds *z_threshold*.

    Parameters
    ----------
    z_threshold:
        Number of standard deviations that triggers an anomaly (default 3).
    window_size:
        Capacity of each metric's rolling window.
    alpha:
        EWMA smoothing factor.
    min_samples:
        Minimum observations before anomaly detection is active.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        window_size: int = 60,
        alpha: float = 0.1,
        min_samples: int = 10,
    ) -> None:
        self.z_threshold = z_threshold
        self.window_size = window_size
        self.alpha = alpha
        self.min_samples = min_samples
        self._windows: Dict[str, RollingWindow] = {}

    def _window(self, metric: str) -> RollingWindow:
        if metric not in self._windows:
            self._windows[metric] = RollingWindow(self.window_size, self.alpha)
        return self._windows[metric]

    def observe(self, metric: str, value: float) -> Optional[Anomaly]:
        """Record a new *value* for *metric* and return an :class:`Anomaly` if detected."""
        win = self._window(metric)
        win.push(value)

        if len(win) < self.min_samples or win.ewma is None:
            return None

        std = win.ewma_std
        if std < 1e-6:
            return None

        z = (value - win.ewma) / std
        if abs(z) >= self.z_threshold:
            return Anomaly(
                metric=metric,
                value=value,
                z_score=z,
                ewma=win.ewma,
                ewma_std=std,
            )
        return None

    def observe_many(self, metrics: Dict[str, float]) -> List[Anomaly]:
        """Observe multiple metrics at once; return all detected anomalies."""
        return [a for m, v in metrics.items() if (a := self.observe(m, v)) is not None]


# ---------------------------------------------------------------------------
# Trend / prediction
# ---------------------------------------------------------------------------


@dataclass
class TrendResult:
    metric: str
    slope: float          # units per second
    predicted_value: float
    horizon_seconds: float
    will_breach: bool
    breach_threshold: float
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        direction = "↑" if self.slope > 0 else "↓"
        breach = f"  ⚠ predicted breach of {self.breach_threshold:.0f}%" if self.will_breach else ""
        return (
            f"Trend for {self.metric!r}: slope={self.slope:+.3f}/s  "
            f"predicted({self.horizon_seconds:.0f}s)={self.predicted_value:.1f}%{direction}{breach}"
        )


class TrendPredictor:
    """OLS linear regression over each metric's rolling window.

    Parameters
    ----------
    horizon_seconds:
        How many seconds ahead to project the current trend.
    window_size:
        Rolling window capacity (each slot represents one poll interval).
    """

    def __init__(self, horizon_seconds: float = 300.0, window_size: int = 30) -> None:
        self.horizon_seconds = horizon_seconds
        self._windows: Dict[str, RollingWindow] = {}
        self._window_size = window_size

    def _window(self, metric: str) -> RollingWindow:
        if metric not in self._windows:
            self._windows[metric] = RollingWindow(self._window_size)
        return self._windows[metric]

    @staticmethod
    def _ols(y: List[float]) -> Tuple[float, float]:
        """Return (slope, intercept) for y indexed by 0..n-1."""
        n = len(y)
        if n < 2:
            return 0.0, y[0] if y else 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        ss_xy = sum((i - x_mean) * (yi - y_mean) for i, yi in enumerate(y))
        slope = ss_xy / ss_xx if ss_xx else 0.0
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def observe(self, metric: str, value: float, poll_interval: float = 30.0) -> Optional[TrendResult]:
        """Record *value* and return a :class:`TrendResult` (once enough data exists)."""
        win = self._window(metric)
        win.push(value)
        if len(win) < 3:
            return None
        slope, intercept = self._ols(win.values)
        steps_ahead = self.horizon_seconds / max(poll_interval, 1.0)
        predicted = intercept + slope * (len(win) - 1 + steps_ahead)
        return TrendResult(
            metric=metric,
            slope=slope / max(poll_interval, 1.0),
            predicted_value=min(max(predicted, 0.0), 100.0),
            horizon_seconds=self.horizon_seconds,
            will_breach=False,  # caller sets this based on threshold
            breach_threshold=0.0,
        )

    def predict_breach(
        self, metric: str, value: float, threshold: float, poll_interval: float = 30.0
    ) -> Optional[TrendResult]:
        """Predict whether *metric* will breach *threshold* within the horizon."""
        result = self.observe(metric, value, poll_interval)
        if result is None:
            return None
        result.will_breach = result.predicted_value >= threshold
        result.breach_threshold = threshold
        return result


# ---------------------------------------------------------------------------
# Problem solver
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    issue: str
    action: str
    priority: int       # 1 = highest
    confidence: float   # 0–1
    source: str = "ml_engine"

    def __str__(self) -> str:
        return f"[P{self.priority} | {self.confidence:.0%}] {self.action}  (issue: {self.issue})"


# Built-in rule table: (issue_keyword, action, base_priority, base_confidence)
_RULES: List[Tuple[str, str, int, float]] = [
    ("cpu", "Identify and terminate CPU-hogging processes", 1, 0.90),
    ("cpu", "Check for runaway background services or update daemons", 2, 0.80),
    ("cpu", "Reduce startup programs to free CPU headroom", 3, 0.70),
    ("memory", "Close unused applications to free RAM", 1, 0.92),
    ("memory", "Check for memory leaks in long-running processes", 2, 0.78),
    ("memory", "Increase swap/virtual memory as a short-term measure", 3, 0.60),
    ("disk", "Delete temporary files and empty the trash", 1, 0.95),
    ("disk", "Move large files to external or cloud storage", 2, 0.80),
    ("disk", "Uninstall unused applications", 3, 0.70),
    ("network", "Check for processes with unexpected high network traffic", 1, 0.85),
    ("process", "Terminate the offending process or reduce its priority", 1, 0.90),
    ("anomaly", "Investigate the anomalous metric for root cause", 1, 0.75),
]


class ProblemSolver:
    """Maps detected issues to prioritised, confidence-ranked recommendations.

    Learns from explicit feedback: call :meth:`feedback` with
    ``accepted=True/False`` to adjust rule confidences over time.

    Parameters
    ----------
    learning_rate:
        How strongly each feedback nudges the confidence score.
    """

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        # Mutable confidence table: key = (keyword, action)
        self._confidence: Dict[Tuple[str, str], float] = {
            (kw, act): conf for kw, act, _, conf in _RULES
        }
        self._rules = _RULES[:]

    def solve(self, issue: str) -> List[Recommendation]:
        """Return a ranked list of :class:`Recommendation` objects for *issue*."""
        issue_lower = issue.lower()
        recs: List[Recommendation] = []
        for kw, action, priority, _ in self._rules:
            if kw in issue_lower:
                conf = self._confidence.get((kw, action), 0.5)
                recs.append(
                    Recommendation(
                        issue=issue,
                        action=action,
                        priority=priority,
                        confidence=conf,
                    )
                )
        # Sort by priority then descending confidence
        recs.sort(key=lambda r: (r.priority, -r.confidence))
        return recs

    def feedback(self, recommendation: Recommendation, accepted: bool) -> None:
        """Update the confidence of a :class:`Recommendation` based on user feedback."""
        key = (
            next((kw for kw, _, _, _ in self._rules if kw in recommendation.issue.lower()), "anomaly"),
            recommendation.action,
        )
        current = self._confidence.get(key, 0.5)
        delta = self.learning_rate if accepted else -self.learning_rate
        self._confidence[key] = max(0.0, min(1.0, current + delta))


# ---------------------------------------------------------------------------
# Façade
# ---------------------------------------------------------------------------


class MLEngine:
    """Unified ML façade used by the :class:`~ai_helper.orchestrator.Orchestrator`.

    Combines anomaly detection, trend prediction and problem solving into a
    single object with a simple ``process(metrics)`` call.

    Parameters
    ----------
    poll_interval:
        Expected seconds between calls to :meth:`process` – used by the
        trend predictor to convert slope from *steps* to *per second*.
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        anomaly_z_threshold: float = 3.0,
        trend_horizon_seconds: float = 300.0,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.detector = AnomalyDetector(z_threshold=anomaly_z_threshold)
        self.predictor = TrendPredictor(horizon_seconds=trend_horizon_seconds)
        self.solver = ProblemSolver()
        self.thresholds: Dict[str, float] = thresholds or {
            "cpu_percent": 85.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
        }

    def process(
        self, metrics: Dict[str, float]
    ) -> Tuple[List[Anomaly], List[TrendResult], List[Recommendation]]:
        """Analyse *metrics*, return (anomalies, trends, recommendations).

        Parameters
        ----------
        metrics:
            Dict mapping metric name to current float value, e.g.
            ``{"cpu_percent": 42.3, "memory_percent": 67.1}``.
        """
        anomalies = self.detector.observe_many(metrics)
        trends: List[TrendResult] = []
        recommendations: List[Recommendation] = []

        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold is not None:
                result = self.predictor.predict_breach(
                    metric, value, threshold, self.poll_interval
                )
                if result is not None:
                    trends.append(result)
                    if result.will_breach:
                        recs = self.solver.solve(metric)
                        recommendations.extend(recs)

        for anomaly in anomalies:
            recs = self.solver.solve(f"anomaly {anomaly.metric}")
            recommendations.extend(recs)

        # Deduplicate recommendations keeping highest confidence copy
        seen: Dict[str, Recommendation] = {}
        for rec in recommendations:
            if rec.action not in seen or rec.confidence > seen[rec.action].confidence:
                seen[rec.action] = rec
        recommendations = sorted(seen.values(), key=lambda r: (r.priority, -r.confidence))

        return anomalies, trends, recommendations
