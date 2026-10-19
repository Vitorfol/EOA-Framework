"""Tests for EOA memory module (SessionStore, InMemory, SQLite, ChatMessage)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eoa.memory import (
    ROLE_AGENT,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    ChatMessage,
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
)


class TestChatMessage:
    """Unit tests for ChatMessage contract validation."""

    def test_valid_chat_message_creation(self) -> None:
        msg = ChatMessage(role=ROLE_USER, content="System overheated.")
        assert msg.role == "user"
        assert msg.content == "System overheated."
        assert isinstance(msg.timestamp, datetime)
        assert msg.timestamp.tzinfo is not None
        assert msg.metadata == {}

    def test_chat_message_custom_role_and_metadata(self) -> None:
        custom_time = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
        msg = ChatMessage(
            role="custom_validator",
            content="Check passed.",
            timestamp=custom_time,
            metadata={"fidelity_score": 0.95, "flags": ["safe"]},
        )
        assert msg.role == "custom_validator"
        assert msg.content == "Check passed."
        assert msg.timestamp == custom_time
        assert msg.metadata["fidelity_score"] == 0.95

    def test_chat_message_empty_role_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="", content="Missing role")

    def test_chat_message_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(
                role=ROLE_ASSISTANT,
                content="Hi",
                unknown_field=123,  # type: ignore[call-arg]
            )

    def test_canonical_role_constants(self) -> None:
        assert ROLE_USER == "user"
        assert ROLE_ASSISTANT == "assistant"
        assert ROLE_SYSTEM == "system"
        assert ROLE_AGENT == "agent"


class TestInMemorySessionStore:
    """Unit tests for InMemorySessionStore."""

    def test_protocol_conformance(self) -> None:
        store = InMemorySessionStore()
        assert isinstance(store, SessionStore)

    def test_append_and_get_history_ordering(self) -> None:
        store = InMemorySessionStore()
        msg1 = ChatMessage(role=ROLE_USER, content="Hello")
        msg2 = ChatMessage(role=ROLE_ASSISTANT, content="Hi there")

        store.append_message("session_1", msg1)
        store.append_message("session_1", msg2)

        history = store.get_history("session_1")
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there"

        # Getting non-existent session returns empty list
        assert store.get_history("unknown_session") == []

    def test_history_returns_defensive_copy(self) -> None:
        store = InMemorySessionStore()
        msg = ChatMessage(role=ROLE_USER, content="Hello")
        store.append_message("s1", msg)

        history = store.get_history("s1")
        history.append(ChatMessage(role=ROLE_ASSISTANT, content="Mutated"))

        assert len(store.get_history("s1")) == 1

    def test_clear_session(self) -> None:
        store = InMemorySessionStore()
        store.append_message("s1", ChatMessage(role=ROLE_USER, content="Msg"))
        assert len(store.get_history("s1")) == 1

        store.clear("s1")
        assert store.get_history("s1") == []
        assert "s1" in store.list_sessions()

    def test_list_and_delete_sessions(self) -> None:
        store = InMemorySessionStore()
        store.append_message("s1", ChatMessage(role=ROLE_USER, content="1"))
        store.append_message("s2", ChatMessage(role=ROLE_USER, content="2"))

        assert set(store.list_sessions()) == {"s1", "s2"}

        store.delete_session("s1")
        assert store.list_sessions() == ["s2"]
        assert store.get_history("s1") == []

    @pytest.mark.asyncio
    async def test_async_methods(self) -> None:
        store = InMemorySessionStore()
        msg = ChatMessage(role=ROLE_USER, content="Async test")

        await store.aappend_message("s_async", msg)
        history = await store.aget_history("s_async")
        assert len(history) == 1
        assert history[0].content == "Async test"

        await store.aclear("s_async")
        assert await store.aget_history("s_async") == []

    def test_export_dict(self) -> None:
        store = InMemorySessionStore()
        store.append_message(
            "s1",
            ChatMessage(role=ROLE_USER, content="Hello", metadata={"k": "v"}),
        )
        exported = store.export_dict("s1")

        assert "s1" in exported
        assert len(exported["s1"]) == 1
        assert exported["s1"][0]["role"] == "user"
        assert exported["s1"][0]["metadata"] == {"k": "v"}


class TestSQLiteSessionStore:
    """Unit tests for SQLiteSessionStore."""

    def test_protocol_conformance(self) -> None:
        store = SQLiteSessionStore(":memory:")
        assert isinstance(store, SessionStore)

    def test_default_db_path_is_eoa_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        store = SQLiteSessionStore()
        assert store.db_path == "eoa.db"
        assert (tmp_path / "eoa.db").exists()

    def test_in_memory_sqlite_basic_operations(self) -> None:
        store = SQLiteSessionStore(":memory:")
        msg1 = ChatMessage(
            role=ROLE_USER,
            content="Telemetry check",
            metadata={"sensor": "PS1"},
        )
        msg2 = ChatMessage(role=ROLE_ASSISTANT, content="Telemetry nominal.")

        store.append_message("sess_100", msg1)
        store.append_message("sess_100", msg2)

        history = store.get_history("sess_100")
        assert len(history) == 2
        assert history[0].role == ROLE_USER
        assert history[0].content == "Telemetry check"
        assert history[0].metadata == {"sensor": "PS1"}
        assert history[1].role == ROLE_ASSISTANT
        assert history[1].content == "Telemetry nominal."

        assert store.list_sessions() == ["sess_100"]

    def test_persistence_across_connection_reopening(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test_sessions.db"

        # Step 1: Open and write
        store1 = SQLiteSessionStore(db_file)
        msg1 = ChatMessage(role=ROLE_USER, content="Initial query")
        store1.append_message("persisted_sess", msg1)

        # Step 2: Open a new instance pointing to same file
        store2 = SQLiteSessionStore(db_file)
        history = store2.get_history("persisted_sess")
        assert len(history) == 1
        assert history[0].content == "Initial query"

        # Step 3: Append more to second store
        msg2 = ChatMessage(
            role=ROLE_AGENT,
            content="Explaining finding",
            metadata={"agent": "Explainer"},
        )
        store2.append_message("persisted_sess", msg2)

        # Step 4: Reopen third instance
        store3 = SQLiteSessionStore(db_file)
        history3 = store3.get_history("persisted_sess")
        assert len(history3) == 2
        assert history3[1].role == ROLE_AGENT
        assert history3[1].metadata == {"agent": "Explainer"}

    def test_clear_session_sqlite(self, tmp_path: Path) -> None:
        db_file = tmp_path / "clear_test.db"
        store = SQLiteSessionStore(db_file)

        store.append_message("s1", ChatMessage(role=ROLE_USER, content="A"))
        store.append_message("s1", ChatMessage(role=ROLE_ASSISTANT, content="B"))
        assert len(store.get_history("s1")) == 2

        store.clear("s1")
        assert store.get_history("s1") == []
        # Session record itself still exists
        assert "s1" in store.list_sessions()

    def test_delete_session_sqlite(self, tmp_path: Path) -> None:
        db_file = tmp_path / "del_test.db"
        store = SQLiteSessionStore(db_file)

        store.append_message("s1", ChatMessage(role=ROLE_USER, content="A"))
        store.append_message("s2", ChatMessage(role=ROLE_USER, content="B"))

        store.delete_session("s1")
        assert "s1" not in store.list_sessions()
        assert store.get_history("s1") == []
        assert "s2" in store.list_sessions()

    def test_validation_logging_and_retrieval(self, tmp_path: Path) -> None:
        db_file = tmp_path / "val_logs.db"
        store = SQLiteSessionStore(db_file)

        store.append_message(
            "sess_val",
            ChatMessage(role=ROLE_USER, content="Why high temp?"),
        )
        store.log_validation(
            session_id="sess_val",
            score=0.91,
            passed=True,
            critique="High fidelity explanation corroborated by SHAP.",
            metadata={"refinements": 0},
        )

        logs = store.get_validation_logs("sess_val")
        assert len(logs) == 1
        assert logs[0]["score"] == 0.91
        assert logs[0]["passed"] is True
        assert "corroborated" in logs[0]["critique"]
        assert logs[0]["metadata"] == {"refinements": 0}

    def test_export_dict_sqlite(self, tmp_path: Path) -> None:
        db_file = tmp_path / "export_test.db"
        store = SQLiteSessionStore(db_file)

        store.append_message("sess_exp", ChatMessage(role=ROLE_USER, content="Query"))
        store.log_validation(
            session_id="sess_exp",
            score=0.88,
            passed=True,
            critique="Passed",
        )

        export = store.export_dict("sess_exp")
        assert len(export["sessions"]) == 1
        assert len(export["messages"]) == 1
        assert len(export["validation_logs"]) == 1
        assert export["messages"][0]["content"] == "Query"
        assert export["validation_logs"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_async_sqlite_methods(self, tmp_path: Path) -> None:
        db_file = tmp_path / "async_test.db"
        store = SQLiteSessionStore(db_file)

        msg = ChatMessage(role=ROLE_USER, content="Async SQLite")
        await store.aappend_message("s_async_sql", msg)

        history = await store.aget_history("s_async_sql")
        assert len(history) == 1
        assert history[0].content == "Async SQLite"

        await store.aclear("s_async_sql")
        assert await store.aget_history("s_async_sql") == []

    def test_auto_directory_creation_for_nested_path(self, tmp_path: Path) -> None:
        nested_db = tmp_path / "subdir" / "deep" / "nested.db"
        assert not nested_db.parent.exists()

        store = SQLiteSessionStore(nested_db)
        store.append_message(
            "nested_sess",
            ChatMessage(role=ROLE_USER, content="Nested DB"),
        )

        assert nested_db.exists()
        assert len(store.get_history("nested_sess")) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not __import__("os").environ.get("GEMINI_API_KEY"),
        reason="Skipping live integration test: GEMINI_API_KEY is not set.",
    )
    async def test_live_gemini_multiturn_sqlite_persistence(
        self, tmp_path: Path
    ) -> None:
        """Integration test: multi-turn chat with real Gemini saved to SQLite."""
        from eoa.llm.gemini import GeminiProvider

        db_file = tmp_path / "live_chat.db"
        store = SQLiteSessionStore(db_file)
        llm = GeminiProvider()

        session_id = "incident_live_01"

        # Turn 1: User asks first question
        q1 = "Respond with exactly the word: PUMP_OVERHEAT"
        await store.aappend_message(session_id, ChatMessage(role=ROLE_USER, content=q1))

        ans1 = await llm.agenerate(q1)
        assert len(ans1.strip()) > 0
        await store.aappend_message(
            session_id,
            ChatMessage(
                role=ROLE_ASSISTANT,
                content=ans1,
                metadata={"turn": 1, "model": llm.model},
            ),
        )

        # Turn 2: Follow-up question
        q2 = "Respond with exactly the word: VALVE_FAILURE"
        await store.aappend_message(session_id, ChatMessage(role=ROLE_USER, content=q2))

        ans2 = await llm.agenerate(q2)
        assert len(ans2.strip()) > 0
        await store.aappend_message(
            session_id,
            ChatMessage(
                role=ROLE_ASSISTANT,
                content=ans2,
                metadata={"turn": 2, "model": llm.model},
            ),
        )

        # Verification: Open a completely new SQLite store instance on the same file
        reopened_store = SQLiteSessionStore(db_file)
        history = reopened_store.get_history(session_id)

        assert len(history) == 4
        assert history[0].role == ROLE_USER
        assert history[0].content == q1
        assert history[1].role == ROLE_ASSISTANT
        assert history[1].content == ans1
        assert history[1].metadata["turn"] == 1
        assert history[2].role == ROLE_USER
        assert history[2].content == q2
        assert history[3].role == ROLE_ASSISTANT
        assert history[3].content == ans2
        assert history[3].metadata["turn"] == 2
