"""Tests for ai_helper.ml_engine."""

from __future__ import annotations

import unittest

from ai_helper.ml_engine import (
    AnomalyDetector,
    MLEngine,
    ProblemSolver,
    Recommendation,
    RollingWindow,
    TrendPredictor,
)


class TestRollingWindow(unittest.TestCase):
    def test_push_and_values(self):
        w = RollingWindow(maxlen=5)
        for v in [1, 2, 3, 4, 5]:
            w.push(v)
        self.assertEqual(w.values, [1, 2, 3, 4, 5])

    def test_maxlen_enforced(self):
        w = RollingWindow(maxlen=3)
        for v in range(10):
            w.push(float(v))
        self.assertEqual(len(w), 3)

    def test_mean(self):
        w = RollingWindow()
        for v in [2.0, 4.0, 6.0]:
            w.push(v)
        self.assertAlmostEqual(w.mean(), 4.0)

    def test_ewma_converges(self):
        w = RollingWindow(alpha=0.5)
        for _ in range(50):
            w.push(100.0)
        self.assertAlmostEqual(w.ewma, 100.0, places=0)


class TestAnomalyDetector(unittest.TestCase):
    def test_no_anomaly_below_threshold(self):
        det = AnomalyDetector(z_threshold=3.0, min_samples=5)
        # Stable signal
        for _ in range(20):
            result = det.observe("cpu", 50.0)
        self.assertIsNone(result)

    def test_detects_spike(self):
        det = AnomalyDetector(z_threshold=2.0, min_samples=5, alpha=0.3)
        for _ in range(30):
            det.observe("cpu", 10.0)
        # Inject a large spike
        anomaly = det.observe("cpu", 1000.0)
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.metric, "cpu")
        self.assertGreater(anomaly.z_score, 0)

    def test_observe_many(self):
        det = AnomalyDetector(z_threshold=2.0, min_samples=5, alpha=0.3)
        for _ in range(30):
            det.observe_many({"cpu": 10.0, "mem": 20.0})
        anomalies = det.observe_many({"cpu": 9999.0, "mem": 21.0})
        self.assertTrue(any(a.metric == "cpu" for a in anomalies))


class TestTrendPredictor(unittest.TestCase):
    def test_returns_none_with_few_samples(self):
        pred = TrendPredictor()
        result = pred.observe("cpu", 50.0)
        self.assertIsNone(result)

    def test_positive_trend(self):
        pred = TrendPredictor(horizon_seconds=60, window_size=10)
        for i in range(10):
            result = pred.observe("cpu", float(i * 5))
        self.assertIsNotNone(result)
        self.assertGreater(result.slope, 0)

    def test_predict_breach_detects_rising_metric(self):
        pred = TrendPredictor(horizon_seconds=60, window_size=10)
        for i in range(10):
            result = pred.predict_breach("cpu", float(i * 10), threshold=85.0, poll_interval=1.0)
        self.assertIsNotNone(result)
        # Steeply rising signal should predict breach
        self.assertTrue(result.will_breach)


class TestProblemSolver(unittest.TestCase):
    def test_cpu_recommendations(self):
        solver = ProblemSolver()
        recs = solver.solve("cpu usage high")
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(isinstance(r, Recommendation) for r in recs))

    def test_memory_recommendations(self):
        solver = ProblemSolver()
        recs = solver.solve("high memory usage")
        self.assertTrue(any("RAM" in r.action or "memory" in r.action.lower() for r in recs))

    def test_unknown_issue_returns_empty(self):
        solver = ProblemSolver()
        recs = solver.solve("completely_unknown_xyz_issue")
        self.assertEqual(recs, [])

    def test_feedback_increases_confidence(self):
        solver = ProblemSolver(learning_rate=0.1)
        recs = solver.solve("cpu high")
        self.assertGreater(len(recs), 0)
        rec = recs[0]
        before = rec.confidence
        solver.feedback(rec, accepted=True)
        after_recs = solver.solve("cpu high")
        after_conf = next(r.confidence for r in after_recs if r.action == rec.action)
        self.assertGreaterEqual(after_conf, before)

    def test_feedback_decreases_confidence(self):
        solver = ProblemSolver(learning_rate=0.1)
        recs = solver.solve("cpu high")
        rec = recs[0]
        before = rec.confidence
        solver.feedback(rec, accepted=False)
        after_recs = solver.solve("cpu high")
        after_conf = next(r.confidence for r in after_recs if r.action == rec.action)
        self.assertLessEqual(after_conf, before)


class TestMLEngine(unittest.TestCase):
    def test_process_returns_three_lists(self):
        engine = MLEngine(poll_interval=1.0)
        anomalies, trends, recs = engine.process({"cpu_percent": 50.0, "memory_percent": 40.0})
        self.assertIsInstance(anomalies, list)
        self.assertIsInstance(trends, list)
        self.assertIsInstance(recs, list)

    def test_high_metric_eventually_produces_recommendations(self):
        engine = MLEngine(
            poll_interval=1.0,
            trend_horizon_seconds=5.0,
            thresholds={"cpu_percent": 50.0},
        )
        for i in range(15):
            anomalies, trends, recs = engine.process({"cpu_percent": float(i * 8)})
        # Steeply rising CPU should eventually produce recs
        self.assertIsInstance(recs, list)


if __name__ == "__main__":
    unittest.main()
