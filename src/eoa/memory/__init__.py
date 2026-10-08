"""Memory module for EOA (Explanation-Oriented Agents).

Provides conversation memory, multi-turn history tracking, and industrial audit logs.
Contains both ephemeral in-memory implementations and persistent SQLite stores.
"""

from __future__ import annotations

from eoa.memory.base import (
    ROLE_AGENT,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    ChatMessage,
    SessionStore,
)
from eoa.memory.in_memory import InMemorySessionStore
from eoa.memory.sqlite import SQLiteSessionStore

__all__ = [
    "ChatMessage",
    "InMemorySessionStore",
    "ROLE_AGENT",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "SessionStore",
    "SQLiteSessionStore",
]
