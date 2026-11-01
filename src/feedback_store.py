"""
Feedback persistence — stores thumb-up/down votes in a local SQLite database.

Schema (minimal):
    feedback(id INTEGER PK, msg_id INTEGER, value TEXT, ts REAL)

No medical content or questions are stored — only the message ID, vote value,
and timestamp.  The database is created automatically on first use.

Usage:
    store = FeedbackStore()          # opens/creates ./data/feedback.db
    store.save(msg_id=42, value="up")
    current = store.load(msg_id=42)  # returns "up", "down", or None
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("MedSource.feedback_store")

_DEFAULT_DB_PATH = "./data/feedback.db"


class FeedbackStore:
    """Thread-safe (single-process) SQLite feedback store."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        try:
            conn = self._connection()
            conn.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_id  INTEGER NOT NULL,
                    value   TEXT    NOT NULL,
                    ts      REAL    NOT NULL
                )"""
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_feedback_msg ON feedback(msg_id)"
            )
            conn.commit()
        except Exception as exc:
            logger.warning("FeedbackStore schema init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, msg_id: int, value: str | None) -> None:
        """Upsert feedback for *msg_id*.

        Passing ``value=None`` removes the feedback entry (toggle off).
        """
        try:
            conn = self._connection()
            if value is None:
                conn.execute("DELETE FROM feedback WHERE msg_id = ?", (msg_id,))
            else:
                conn.execute(
                    """INSERT INTO feedback(msg_id, value, ts) VALUES(?, ?, ?)
                       ON CONFLICT(msg_id) DO UPDATE SET value=excluded.value, ts=excluded.ts""",
                    (msg_id, value, time.time()),
                )
            conn.commit()
        except Exception as exc:
            logger.warning("FeedbackStore.save failed (msg_id=%s): %s", msg_id, exc)

    def load(self, msg_id: int) -> str | None:
        """Return the stored feedback for *msg_id*, or ``None`` if absent."""
        try:
            conn = self._connection()
            row = conn.execute(
                "SELECT value FROM feedback WHERE msg_id = ?", (msg_id,)
            ).fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.warning("FeedbackStore.load failed (msg_id=%s): %s", msg_id, exc)
            return None

    def load_all(self) -> dict[int, str]:
        """Return all stored feedback as {msg_id: value}."""
        try:
            conn = self._connection()
            rows = conn.execute("SELECT msg_id, value FROM feedback").fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as exc:
            logger.warning("FeedbackStore.load_all failed: %s", exc)
            return {}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
