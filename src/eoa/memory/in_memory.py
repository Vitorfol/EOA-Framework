"""In-memory implementation of the SessionStore protocol.

Thread-safe, volatile memory store tailored for prototyping, unit testing,
and stateless microservices without disk persistence.
"""

from __future__ import annotations

import threading
from typing import Any

from eoa.memory.base import ChatMessage, SessionStore


class InMemorySessionStore(SessionStore):
    """A thread-safe, in-memory implementation of SessionStore.

    Stores conversation turns in an internal dictionary indexed by session ID.
    Thread safety is guaranteed via a re-entrant lock (threading.RLock).

    Example:
        >>> store = InMemorySessionStore()
        >>> store.append_message("s-1", ChatMessage(role="user", content="Hello"))
        >>> len(store.get_history("s-1"))
        1
        >>> store.clear("s-1")
        >>> store.get_history("s-1")
        []
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory session store."""
        self._lock = threading.RLock()
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve recorded messages for a session in chronological order.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            A list containing copies of the ChatMessage objects.
        """
        with self._lock:
            # Return a shallow copy of the list to prevent external mutation
            return list(self._sessions.get(session_id, []))

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a new message turn to the specified session history.

        Args:
            session_id: Unique identifier for the conversation session.
            message: The ChatMessage instance to store.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append(message)

    def clear(self, session_id: str) -> None:
        """Clear all messages from the session while preserving the entry.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()

    def list_sessions(self) -> list[str]:
        """List all active session identifiers stored in memory.

        Returns:
            List of session identifier strings.
        """
        with self._lock:
            return list(self._sessions.keys())

    def delete_session(self, session_id: str) -> None:
        """Completely purge a session and its associated messages from memory.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        with self._lock:
            self._sessions.pop(session_id, None)
            self._metadata.pop(session_id, None)

    def get_session_metadata(self, session_id: str) -> dict[str, Any]:
        """Retrieve stored session metadata dictionary.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            Session metadata dictionary (or empty dict if unset).
        """
        with self._lock:
            return dict(self._metadata.get(session_id, {}))

    def set_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Store or update session metadata dictionary in memory.

        Args:
            session_id: Unique identifier for the conversation session.
            metadata: Metadata dictionary to persist for the session.
        """
        with self._lock:
            self._metadata[session_id] = dict(metadata)

    async def aget_history(self, session_id: str) -> list[ChatMessage]:
        """Asynchronously retrieve all messages for a given session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.
        """
        return self.get_history(session_id)

    async def aappend_message(self, session_id: str, message: ChatMessage) -> None:
        """Asynchronously append a new message turn to the session history.

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

    def export_dict(
        self, session_id: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Export session data as plain dictionaries for debugging.

        Args:
            session_id: Optional session ID to export. If None, exports all.

        Returns:
            Dictionary mapping session IDs to lists of message dictionaries.
        """
        with self._lock:
            if session_id is not None:
                messages = self._sessions.get(session_id, [])
                return {session_id: [m.model_dump(mode="json") for m in messages]}
            return {
                sid: [m.model_dump(mode="json") for m in msgs]
                for sid, msgs in self._sessions.items()
            }
