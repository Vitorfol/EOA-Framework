"""Persistent SQLite implementation of the SessionStore protocol.

Lightweight, robust relational storage using Python's standard library `sqlite3`.
Zero external database dependencies. Designed for enterprise compliance,
industrial auditability, and high developer convenience.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eoa.memory.base import ChatMessage, SessionStore


class SQLiteSessionStore(SessionStore):
    """A persistent, relational implementation of SessionStore backed by SQLite.

    Features:
    - Zero dependencies: Uses Python's native `sqlite3`.
    - Auto-DDL: Automatically initializes database tables on instantiation.
    - Concurrency friendly: WAL journal mode and busy timeout configured.
    - Audit ready: Stores structured messages with JSON metadata and validation logs.
    - Developer utilities: Methods for export and diagnostic inspection.

    Args:
        db_path: Path to the SQLite DB file or ":memory:" for in-memory SQLite DB.
        timeout: Lock acquisition timeout in seconds (default: 10.0).

    Example:
        >>> store = SQLiteSessionStore("conversations.db")
        >>> msg = ChatMessage(role="user", content="Explain pump fail")
        >>> store.append_message("plant-01", msg)
        >>> len(store.get_history("plant-01"))
        1
    """

    def __init__(self, db_path: str | Path = "eoa.db", timeout: float = 10.0) -> None:
        """Initialize SQLiteSessionStore, defaulting to 'eoa.db'."""
        self.db_path = (
            str(db_path) if isinstance(db_path, (str, Path)) else str(db_path)
        )
        self.timeout = timeout
        self._lock = threading.RLock()

        self._mem_conn: sqlite3.Connection | None = None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # For :memory:, keep a persistent connection alive so data persists
            self._mem_conn = sqlite3.connect(
                ":memory:",
                timeout=self.timeout,
                check_same_thread=False,
                isolation_level=None,
            )
            self._mem_conn.row_factory = sqlite3.Row
            with self._mem_conn:
                self._mem_conn.execute("PRAGMA foreign_keys = ON;")

        self._init_db()

    def _init_db(self) -> None:
        """Create database schema if it doesn't already exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_id
        ON messages(session_id, timestamp);

        CREATE TABLE IF NOT EXISTS validation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_id INTEGER,
            score REAL NOT NULL,
            passed INTEGER NOT NULL,
            critique TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_validation_logs_session_id
        ON validation_logs(session_id);
        """
        with self._connect() as conn:
            with self._lock:
                conn.executescript(schema)

    def _connect(self) -> sqlite3.Connection:
        """Return the active connection, creating a configured one if not in-memory."""
        if self._mem_conn is not None:
            return self._mem_conn

        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        """Close connection if it is not the persistent in-memory connection."""
        if self._mem_conn is None:
            conn.close()

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a message turn to the specified session history in SQLite.

        If the session does not exist, it will be automatically created.

        Args:
            session_id: Unique identifier for the conversation session.
            message: The ChatMessage instance to store.
        """
        now_iso = datetime.now(UTC).isoformat()
        msg_timestamp_iso = message.timestamp.isoformat()
        metadata_json = json.dumps(message.metadata or {}, ensure_ascii=False)

        conn = self._connect()
        try:
            with self._lock:
                with conn:
                    # Upsert session
                    conn.execute(
                        """
                        INSERT INTO sessions (id, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, '{}')
                        ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                        """,
                        (session_id, now_iso, now_iso),
                    )
                    # Insert message
                    conn.execute(
                        """
                        INSERT INTO messages
                        (session_id, role, content, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            message.role,
                            message.content,
                            msg_timestamp_iso,
                            metadata_json,
                        ),
                    )
        finally:
            self._close_conn(conn)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve all recorded messages for a session, ordered chronologically.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.
        """
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    """
                    SELECT role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
        finally:
            self._close_conn(conn)

        result: list[ChatMessage] = []
        for row in rows:
            ts = datetime.fromisoformat(row["timestamp"])
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            result.append(
                ChatMessage(
                    role=row["role"],
                    content=row["content"],
                    timestamp=ts,
                    metadata=meta,
                )
            )
        return result

    def clear(self, session_id: str) -> None:
        """Clear all messages from the session while preserving the session record.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        now_iso = datetime.now(UTC).isoformat()
        conn = self._connect()
        try:
            with self._lock:
                with conn:
                    conn.execute(
                        "DELETE FROM messages WHERE session_id = ?",
                        (session_id,),
                    )
                    conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE id = ?",
                        (now_iso, session_id),
                    )
        finally:
            self._close_conn(conn)

    def list_sessions(self) -> list[str]:
        """List all session IDs currently recorded in the SQLite database.

        Returns:
            List of session identifier strings.
        """
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    "SELECT id FROM sessions ORDER BY updated_at DESC"
                )
                rows = cursor.fetchall()
                return [row["id"] for row in rows]
        finally:
            self._close_conn(conn)

    def delete_session(self, session_id: str) -> None:
        """Completely delete a session and all its associated messages and logs.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        conn = self._connect()
        try:
            with self._lock:
                with conn:
                    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        finally:
            self._close_conn(conn)

    async def aget_history(self, session_id: str) -> list[ChatMessage]:
        """Asynchronously retrieve all messages for a given session.

        Delegates to get_history.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.
        """
        return self.get_history(session_id)

    async def aappend_message(self, session_id: str, message: ChatMessage) -> None:
        """Asynchronously append a message turn to SQLite.

        Args:
            session_id: Unique identifier for the conversation session.
            message: The ChatMessage instance to store.
        """
        self.append_message(session_id, message)

    async def aclear(self, session_id: str) -> None:
        """Asynchronously clear all messages from the session history.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        self.clear(session_id)

    def log_validation(
        self,
        session_id: str,
        score: float,
        passed: bool,
        critique: str | None = None,
        message_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a validation audit entry for compliance, evaluation, and ICSE metrics.

        Args:
            session_id: The session being validated.
            score: Numeric fidelity or validation score (0.0 to 1.0).
            passed: Whether the explanation passed the quality gate.
            critique: Optional critique notes or feedback from ValidatorAgent.
            message_id: Optional ID of the specific message being evaluated.
            metadata: Optional extra metadata dictionary.
        """
        now_iso = datetime.now(UTC).isoformat()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        conn = self._connect()
        try:
            with self._lock:
                with conn:
                    # Ensure session exists
                    conn.execute(
                        """
                        INSERT INTO sessions (id, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, '{}')
                        ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                        """,
                        (session_id, now_iso, now_iso),
                    )
                    conn.execute(
                        """
                        INSERT INTO validation_logs
                        (
                            session_id, message_id, score, passed,
                            critique, timestamp, metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            message_id,
                            float(score),
                            1 if passed else 0,
                            critique,
                            now_iso,
                            metadata_json,
                        ),
                    )
        finally:
            self._close_conn(conn)

    def get_validation_logs(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all validation logs recorded for a given session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of dictionaries representing validation records.
        """
        conn = self._connect()
        try:
            with self._lock:
                cursor = conn.execute(
                    """
                    SELECT
                        id, session_id, message_id, score,
                        passed, critique, timestamp, metadata
                    FROM validation_logs
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
        finally:
            self._close_conn(conn)

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "message_id": row["message_id"],
                "score": row["score"],
                "passed": bool(row["passed"]),
                "critique": row["critique"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ]

    def export_dict(self, session_id: str | None = None) -> dict[str, Any]:
        """Export session, message, and validation data as structured dictionaries.

        Args:
            session_id: Optional session ID to filter by. If None, exports entire DB.

        Returns:
            A dictionary containing lists of sessions, messages, and validation logs.
        """
        conn = self._connect()
        try:
            with self._lock:
                if session_id is not None:
                    s_cur = conn.execute(
                        "SELECT * FROM sessions WHERE id = ?", (session_id,)
                    )
                    m_cur = conn.execute(
                        "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                        (session_id,),
                    )
                    v_cur = conn.execute(
                        "SELECT * FROM validation_logs "
                        "WHERE session_id = ? ORDER BY id ASC",
                        (session_id,),
                    )
                else:
                    s_cur = conn.execute(
                        "SELECT * FROM sessions ORDER BY updated_at DESC"
                    )
                    m_cur = conn.execute("SELECT * FROM messages ORDER BY id ASC")
                    v_cur = conn.execute(
                        "SELECT * FROM validation_logs ORDER BY id ASC"
                    )

                sessions = [dict(row) for row in s_cur.fetchall()]
                messages = [dict(row) for row in m_cur.fetchall()]
                validation_logs = [dict(row) for row in v_cur.fetchall()]
        finally:
            self._close_conn(conn)

        # Parse JSON columns for clean output
        for s in sessions:
            if "metadata" in s and s["metadata"]:
                s["metadata"] = json.loads(s["metadata"])
        for m in messages:
            if "metadata" in m and m["metadata"]:
                m["metadata"] = json.loads(m["metadata"])
        for v in validation_logs:
            v["passed"] = bool(v["passed"])
            if "metadata" in v and v["metadata"]:
                v["metadata"] = json.loads(v["metadata"])

        return {
            "sessions": sessions,
            "messages": messages,
            "validation_logs": validation_logs,
        }
