"""Persistent memory for AI Helper.

Stores everything AI Helper learns over time in a local SQLite database so
the assistant gets smarter the longer it runs:

* **Anomalies** — past CPU/memory/GPU spikes with timestamps
* **Agent conversations** — goals the user gave and the answers produced
* **File access patterns** — which files the user reads/writes most
* **AI model usage** — which Ollama models are invoked and how often
* **User preferences** — key/value settings the user or agent can update

The database lives at ``<INSTALL_DIR>/memory.db`` by default so it is
always on the D drive.

Usage
-----
::

    from ai_helper.memory import Memory

    mem = Memory()

    # Store a preference
    mem.set_preference("ollama_model", "llama3")

    # Record an anomaly
    mem.record_anomaly("cpu", value=97.3, z_score=4.1)

    # Record an agent conversation
    mem.record_conversation("What uses the most CPU?", "chrome.exe at 45%")

    # Query recent anomalies
    for row in mem.recent_anomalies(limit=5):
        print(row)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS anomalies (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    metric    TEXT    NOT NULL,
    value     REAL    NOT NULL,
    z_score   REAL    NOT NULL,
    details   TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    goal      TEXT    NOT NULL,
    answer    TEXT    NOT NULL,
    steps     INTEGER NOT NULL DEFAULT 0,
    model     TEXT
);

CREATE TABLE IF NOT EXISTS file_access (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    path      TEXT    NOT NULL,
    operation TEXT    NOT NULL   -- 'read' | 'write' | 'search'
);

CREATE TABLE IF NOT EXISTS model_usage (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    model     TEXT    NOT NULL,
    prompt_chars  INTEGER NOT NULL DEFAULT 0,
    response_chars INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS preferences (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_ts REAL NOT NULL
);
"""


@dataclass
class AnomalyRecord:
    id: int
    ts: float
    metric: str
    value: float
    z_score: float
    details: str

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))

    def __str__(self) -> str:
        return (f"[{self.time_str}] {self.metric} = {self.value:.1f}  "
                f"(z={self.z_score:.1f})  {self.details}")


@dataclass
class ConversationRecord:
    id: int
    ts: float
    goal: str
    answer: str
    steps: int
    model: str

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))

    def __str__(self) -> str:
        return f"[{self.time_str}] Q: {self.goal[:80]}  A: {self.answer[:80]}"


class Memory:
    """SQLite-backed persistent memory store.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to ``<INSTALL_DIR>/memory.db``.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            from . import config as _cfg  # noqa: PLC0415
            db_path = _cfg.get_install_dir() / "memory.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Context manager / connection
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Anomalies
    # ------------------------------------------------------------------

    def record_anomaly(
        self,
        metric: str,
        value: float,
        z_score: float,
        details: str = "",
    ) -> None:
        """Save a detected anomaly."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO anomalies (ts, metric, value, z_score, details) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), metric, value, z_score, details),
            )

    def recent_anomalies(self, limit: int = 20, metric: Optional[str] = None) -> List[AnomalyRecord]:
        """Return the most recent anomalies, newest first."""
        with self._conn() as conn:
            if metric:
                rows = conn.execute(
                    "SELECT * FROM anomalies WHERE metric=? ORDER BY ts DESC LIMIT ?",
                    (metric, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM anomalies ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
        return [AnomalyRecord(**dict(r)) for r in rows]

    def anomaly_count(self, metric: Optional[str] = None, since: float = 0.0) -> int:
        """Return total number of recorded anomalies (optionally filtered)."""
        with self._conn() as conn:
            if metric:
                return conn.execute(
                    "SELECT COUNT(*) FROM anomalies WHERE metric=? AND ts>=?",
                    (metric, since),
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM anomalies WHERE ts>=?", (since,)
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def record_conversation(
        self,
        goal: str,
        answer: str,
        steps: int = 0,
        model: str = "",
    ) -> None:
        """Save an agent conversation."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (ts, goal, answer, steps, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), goal, answer, steps, model),
            )

    def recent_conversations(self, limit: int = 10) -> List[ConversationRecord]:
        """Return the most recent conversations, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [ConversationRecord(**dict(r)) for r in rows]

    def search_conversations(self, keyword: str) -> List[ConversationRecord]:
        """Full-text search across goal and answer fields."""
        kw = f"%{keyword.lower()}%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations "
                "WHERE lower(goal) LIKE ? OR lower(answer) LIKE ? "
                "ORDER BY ts DESC LIMIT 20",
                (kw, kw),
            ).fetchall()
        return [ConversationRecord(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # File access patterns
    # ------------------------------------------------------------------

    def record_file_access(self, path: str, operation: str = "read") -> None:
        """Record that a file was accessed."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO file_access (ts, path, operation) VALUES (?, ?, ?)",
                (time.time(), path, operation),
            )

    def frequent_files(self, limit: int = 10, operation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return files accessed most often, with access counts."""
        with self._conn() as conn:
            if operation:
                rows = conn.execute(
                    "SELECT path, COUNT(*) as count FROM file_access "
                    "WHERE operation=? GROUP BY path ORDER BY count DESC LIMIT ?",
                    (operation, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT path, COUNT(*) as count FROM file_access "
                    "GROUP BY path ORDER BY count DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Model usage
    # ------------------------------------------------------------------

    def record_model_usage(
        self,
        model: str,
        prompt_chars: int = 0,
        response_chars: int = 0,
    ) -> None:
        """Record an Ollama (or other LLM) inference call."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO model_usage (ts, model, prompt_chars, response_chars) "
                "VALUES (?, ?, ?, ?)",
                (time.time(), model, prompt_chars, response_chars),
            )

    def model_stats(self) -> List[Dict[str, Any]]:
        """Return usage statistics grouped by model."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model, COUNT(*) as calls, "
                "SUM(prompt_chars) as total_prompt_chars, "
                "SUM(response_chars) as total_response_chars "
                "FROM model_usage GROUP BY model ORDER BY calls DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Store a user/agent preference."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO preferences (key, value, updated_ts) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_ts=excluded.updated_ts",
                (key, json.dumps(value), time.time()),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored preference, or *default* if not set."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    def all_preferences(self) -> Dict[str, Any]:
        """Return all stored preferences as a dict."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        result: Dict[str, Any] = {}
        for row in rows:
            try:
                result[row[0]] = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                result[row[0]] = row[1]
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable memory summary."""
        n_anomalies = self.anomaly_count()
        conversations = self.recent_conversations(limit=3)
        prefs = self.all_preferences()
        frequent = self.frequent_files(limit=3)
        model_st = self.model_stats()

        lines = [f"=== AI Helper Memory ({self.db_path.name}) ==="]
        lines.append(f"  Anomalies recorded  : {n_anomalies}")
        lines.append(f"  Conversations stored: {len(self.recent_conversations(limit=9999))}")
        if prefs:
            lines.append("  Preferences:")
            for k, v in prefs.items():
                lines.append(f"    {k} = {v!r}")
        if frequent:
            lines.append("  Most accessed files:")
            for f in frequent:
                lines.append(f"    {f['count']}x  {f['path']}")
        if model_st:
            lines.append("  Model usage:")
            for m in model_st:
                lines.append(f"    {m['model']}: {m['calls']} calls")
        if conversations:
            lines.append("  Recent conversations:")
            for c in conversations:
                lines.append(f"    {c}")
        return "\n".join(lines)
