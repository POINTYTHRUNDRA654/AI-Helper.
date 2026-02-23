"""Tests for ai_helper.memory."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_helper.memory import Memory, AnomalyRecord, ConversationRecord


class TestMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db = Path(self._tmp.name) / "test_memory.db"
        self.mem = Memory(db_path=self.db)

    def tearDown(self):
        self._tmp.cleanup()

    # --- Anomalies ---

    def test_record_and_retrieve_anomaly(self):
        self.mem.record_anomaly("cpu", value=97.5, z_score=4.1, details="spike")
        records = self.mem.recent_anomalies()
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r.metric, "cpu")
        self.assertAlmostEqual(r.value, 97.5)
        self.assertAlmostEqual(r.z_score, 4.1)

    def test_anomaly_count(self):
        for _ in range(5):
            self.mem.record_anomaly("mem", value=90.0, z_score=3.0)
        self.assertEqual(self.mem.anomaly_count(), 5)
        self.assertEqual(self.mem.anomaly_count(metric="mem"), 5)
        self.assertEqual(self.mem.anomaly_count(metric="cpu"), 0)

    def test_anomaly_count_since(self):
        self.mem.record_anomaly("cpu", 90.0, 3.0)
        future = time.time() + 10
        self.assertEqual(self.mem.anomaly_count(since=future), 0)

    def test_anomaly_str(self):
        self.mem.record_anomaly("gpu", value=95.0, z_score=5.0, details="hot")
        rec = self.mem.recent_anomalies()[0]
        text = str(rec)
        self.assertIn("gpu", text)
        self.assertIn("95", text)

    def test_recent_anomalies_limit(self):
        for i in range(10):
            self.mem.record_anomaly("cpu", float(i), float(i))
        results = self.mem.recent_anomalies(limit=3)
        self.assertEqual(len(results), 3)

    def test_recent_anomalies_metric_filter(self):
        self.mem.record_anomaly("cpu", 90.0, 3.0)
        self.mem.record_anomaly("mem", 85.0, 2.5)
        cpu = self.mem.recent_anomalies(metric="cpu")
        self.assertEqual(len(cpu), 1)
        self.assertEqual(cpu[0].metric, "cpu")

    # --- Conversations ---

    def test_record_and_retrieve_conversation(self):
        self.mem.record_conversation("What is my CPU?", "CPU is 45%", steps=1, model="llama3")
        convos = self.mem.recent_conversations()
        self.assertEqual(len(convos), 1)
        c = convos[0]
        self.assertEqual(c.goal, "What is my CPU?")
        self.assertEqual(c.answer, "CPU is 45%")
        self.assertEqual(c.model, "llama3")

    def test_conversation_str(self):
        self.mem.record_conversation("goal", "answer")
        c = self.mem.recent_conversations()[0]
        self.assertIn("goal", str(c))

    def test_search_conversations(self):
        self.mem.record_conversation("CPU spike detected", "Spike was brief", steps=2)
        self.mem.record_conversation("Disk usage high", "Clean up old files")
        results = self.mem.search_conversations("cpu")
        self.assertEqual(len(results), 1)
        self.assertIn("CPU", results[0].goal)

    def test_recent_conversations_limit(self):
        for i in range(15):
            self.mem.record_conversation(f"goal {i}", f"answer {i}")
        results = self.mem.recent_conversations(limit=5)
        self.assertEqual(len(results), 5)

    # --- File access ---

    def test_record_and_retrieve_file_access(self):
        self.mem.record_file_access("/home/user/notes.txt", "read")
        self.mem.record_file_access("/home/user/notes.txt", "read")
        self.mem.record_file_access("/home/user/other.py", "write")
        frequent = self.mem.frequent_files(limit=5)
        self.assertEqual(frequent[0]["path"], "/home/user/notes.txt")
        self.assertEqual(frequent[0]["count"], 2)

    def test_frequent_files_operation_filter(self):
        self.mem.record_file_access("/a.txt", "read")
        self.mem.record_file_access("/b.txt", "write")
        writes = self.mem.frequent_files(operation="write")
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["path"], "/b.txt")

    # --- Model usage ---

    def test_record_and_retrieve_model_usage(self):
        self.mem.record_model_usage("llama3", prompt_chars=100, response_chars=200)
        self.mem.record_model_usage("llama3", prompt_chars=50, response_chars=150)
        self.mem.record_model_usage("mistral", prompt_chars=80, response_chars=120)
        stats = self.mem.model_stats()
        models = {s["model"]: s for s in stats}
        self.assertEqual(models["llama3"]["calls"], 2)
        self.assertEqual(models["mistral"]["calls"], 1)

    # --- Preferences ---

    def test_set_and_get_preference(self):
        self.mem.set_preference("ollama_model", "llama3")
        self.assertEqual(self.mem.get_preference("ollama_model"), "llama3")

    def test_preference_default(self):
        self.assertIsNone(self.mem.get_preference("nonexistent"))
        self.assertEqual(self.mem.get_preference("nonexistent", "default"), "default")

    def test_preference_overwrite(self):
        self.mem.set_preference("theme", "dark")
        self.mem.set_preference("theme", "light")
        self.assertEqual(self.mem.get_preference("theme"), "light")

    def test_preference_complex_value(self):
        self.mem.set_preference("thresholds", {"cpu": 85, "mem": 90})
        val = self.mem.get_preference("thresholds")
        self.assertEqual(val["cpu"], 85)

    def test_all_preferences(self):
        self.mem.set_preference("a", 1)
        self.mem.set_preference("b", "two")
        prefs = self.mem.all_preferences()
        self.assertIn("a", prefs)
        self.assertIn("b", prefs)

    # --- Summary ---

    def test_summary_contains_key_info(self):
        self.mem.record_anomaly("cpu", 97.0, 4.0)
        self.mem.record_conversation("hello", "world")
        self.mem.set_preference("voice", True)
        summary = self.mem.summary()
        self.assertIn("Anomalies", summary)
        self.assertIn("Preferences", summary)


if __name__ == "__main__":
    unittest.main()
