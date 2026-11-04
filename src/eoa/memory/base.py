"""Core data structures and protocols for conversation memory.

This module defines the fundamental contracts for multi-turn sessions in EOA:
- ChatMessage: Validated container for dialogue turns, role attribution, and metadata.
- SessionStore: The canonical protocol that any memory backend must fulfill.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# Canonical role constants for standard dialogue participants
ROLE_USER: str = "user"
ROLE_ASSISTANT: str = "assistant"
ROLE_SYSTEM: str = "system"
ROLE_AGENT: str = "agent"


class ChatMessage(BaseModel):
    """Represents an individual message turn within an EOA session.

    Attributes:
        role: Originator role (e.g. 'user', 'assistant', 'system', 'agent').
        content: Textual payload of the message.
        timestamp: UTC timestamp when the turn occurred. Defaults to UTC now.
        metadata: Optional key-value dictionary for contextual information,
            such as validation scores, agent identifiers, or latency measurements.

    Example:
        >>> msg = ChatMessage(role="user", content="Why did PS1 overheat?")
        >>> msg.role
        'user'
        >>> isinstance(msg.timestamp, datetime)
        True
        >>> msg.metadata
        {}
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    role: str = Field(
        ...,
        min_length=1,
        description="The sender role (e.g., 'user', 'assistant', 'system', 'agent')",
    )
    content: str = Field(
        ...,
        description="The message textual content",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC datetime when the message was recorded",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extensible metadata for auditing, metrics, or telemetry",
    )


@runtime_checkable
class SessionStore(Protocol):
    """Protocol defining the storage contract for conversational history.

    All implementations (e.g., InMemorySessionStore, SQLiteSessionStore) must satisfy
    this interface to be pluggable across EOA explanation and chat orchestration layers.
    """

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve all recorded messages for a session, ordered chronologically.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.
        """
        ...

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a new message turn to the specified session history.

        Args:
            session_id: Unique identifier for the conversation session.
            message: The ChatMessage instance to store.
        """
        ...

    def clear(self, session_id: str) -> None:
        """Clear all messages from the session while preserving the session ID.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        ...

    def list_sessions(self) -> list[str]:
        """List all active session identifiers managed by this store.

        Returns:
            List of session identifier strings.
        """
        ...

    def get_session_metadata(self, session_id: str) -> dict[str, Any]:
        """Retrieve stored session metadata dictionary.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            Session metadata dictionary (or empty dict if unset).
        """
        ...

    def set_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Store or update session metadata dictionary.

        Args:
            session_id: Unique identifier for the conversation session.
            metadata: Metadata dictionary to persist for the session.
        """
        ...

    def delete_session(self, session_id: str) -> None:
        """Completely purge a session and its associated messages.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        ...

    async def aget_history(self, session_id: str) -> list[ChatMessage]:
        """Asynchronously retrieve all messages for a given session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.
        """
        ...

    async def aappend_message(self, session_id: str, message: ChatMessage) -> None:
        """Asynchronously append a new message turn to the session history.

        Args:
            session_id: Unique identifier for the conversation session.
            message: The ChatMessage instance to store.
        """
        ...

    async def aclear(self, session_id: str) -> None:
        """Asynchronously clear all messages from the session history.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        ...
