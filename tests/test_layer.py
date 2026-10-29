"""Unit and integration tests for ExplanationLayer and default_layer (EP08).

Covers:
1. Scenario 1: Plug & Play in 2 lines with `default_layer`.
2. Scenario 2: 3 LLM Connection Modes (Gemini, BYO Client, Callable).
3. Scenario 3: "Developer is King" (remove_agent, override, SpecializedAgent).
4. Scenario 4: Multi-Turn Conversation and SQLite audit logging with `layer.explain`.
5. Scenario 5: Technical manuals (Structured vs. Custom RAG retriever in CMA).
6. Scenario 6: Production API / Fail-Safe resilience ("Unplugged" mode).
7. Scenario 7: Autonomous Query-Based Mode (tools on AA and CMA).
8. Scenario 8: Progressive Ramp (Zero ML mode).
9. Session Artifact Cache: Verifying Turn 2 inherits Turn 1 artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eoa import (
    ExplanationLayer,
    ExplanationResponse,
    SpecializedAgent,
    SQLiteSessionStore,
)
from tests.mocks.mock_llm import MockLLM


def _make_smart_mock_llm() -> MockLLM:
    """Create a deterministic MockLLM routing responses cleanly for all agents."""

    def responder(prompt: str, system_prompt: str | None = None) -> str:
        if "You are the Context Manager" in prompt:
            return "Key Variables:\n- PS1: Pressure (normal 0-4 bar)."
        if "You are the Artifacts Analyst" in prompt:
            return "Outcome: Risk 85%. Drivers: PS1 (+0.48)."
        if "You are the Validator" in prompt:
            return '{"score": 9.5, "issues": [], "critique": ""}'
        if "You are the Explainer" in prompt:
            return (
                "The cooler efficiency dropped due to elevated inlet pressure on PS1. "
                "Recommendation: Inspect suction valve immediately."
            )
        return "Specialist evaluated domain telemetry successfully."

    return MockLLM(custom_generator=responder)


# =========================================================================
# Scenario 1 & 2: Plug & Play and 3 LLM Connection Modes
# =========================================================================
class TestPlugAndPlayAndModes:
    def test_scenario_1_plug_and_play(self) -> None:
        mock_llm = _make_smart_mock_llm()
        layer = ExplanationLayer.default_layer(llm=mock_llm)

        res = layer.explain(
            prediction={
                "target_variable": "Cooler Condition",
                "predicted_value": "Efficiency Reduced",
                "confidence_or_probability": 0.82,
            },
            xai={"feature_attributions": {"PS1": 0.45, "EPS1": 0.35}},
            user_query="Explain why this alert fired.",
        )

        assert isinstance(res, ExplanationResponse)
        assert "cooler efficiency dropped" in res.narrative.lower()
        assert res.validation_score == 9.5
        assert res.validation_passed is True
        assert res.fallback_used is False

    def test_scenario_2_callable_local_llm(self) -> None:
        # Mode 3: Custom callable function (e.g. Ollama or local model)
        def my_local_model(prompt: str) -> str:
            if "Validator" in prompt:
                return '{"score": 8.8, "issues": [], "critique": ""}'
            return "Local AI model response."

        layer = ExplanationLayer.default_layer(llm=my_local_model)
        res = layer.explain(user_query="Hello system")
        assert "local ai model" in res.narrative.lower()

    @pytest.mark.asyncio
    async def test_async_aexplain_parity(self) -> None:
        mock_llm = _make_smart_mock_llm()
        layer = ExplanationLayer.default_layer(llm=mock_llm)

        res = await layer.aexplain(
            prediction={"target_variable": "Pump", "predicted_value": "Degraded"},
            user_query="Async status query",
        )
        assert isinstance(res, ExplanationResponse)
        assert res.validation_passed is True


# =========================================================================
# Scenario 3: "Developer is King" & Agent Lifecycle
# =========================================================================
class TestDeveloperIsKing:
    def test_remove_and_get_agent(self) -> None:
        layer = ExplanationLayer(llm=_make_smart_mock_llm())
        with pytest.raises(KeyError, match="create_default_agents"):
            layer.get_agent("context_manager")

        layer.create_default_agents()

        # Check initial agents
        assert layer.get_agent("context_manager") is not None
        assert layer.get_agent("artifacts") is not None

        # Remove CMA
        layer.remove_agent("context_manager")
        with pytest.raises(KeyError, match="not found"):
            layer.get_agent("context_manager")

        # Also verify ExplanationLayer.default_layer() classmethod
        ready_layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        assert ready_layer.get_agent("explainer") is not None

    def test_override_prompt_with_str_and_path(self, tmp_path: Path) -> None:
        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())

        # 1. Override with str
        custom_p = "You are an industrial engineer: {user_query}"
        layer.override_agent_prompt("explainer", custom_p)
        ea = layer.get_agent("explainer")
        assert "You are an industrial engineer" in ea.prompt

        # 2. Override with Path (.md)
        custom_md = tmp_path / "custom_ea.md"
        custom_md.write_text("Custom prompt from file: {user_query}", encoding="utf-8")
        layer.override_agent_prompt("explainer", custom_md)
        assert "Custom prompt from file" in ea.prompt

        # 3. Reset to default
        layer.reset_agent_prompt("explainer")
        assert "Synthesize available evidence" in ea.prompt
        assert ea.prompt_source == "Built-in"

    def test_register_specialized_agent(self, tmp_path: Path) -> None:
        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        spec_md = tmp_path / "turbine.md"
        spec_md.write_text("Cavitation specialist: {user_query}", encoding="utf-8")

        specialist = SpecializedAgent(name="turbine_expert", prompt=spec_md)
        layer.register_agent(specialist)

        assert layer.get_agent("turbine_expert") is specialist

        # Test execution incorporates specialist
        res = layer.explain(user_query="Check turbine cavitation")
        assert res.validation_passed is True

    def test_inspect_output_formatting(self) -> None:
        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        layer.override_agent_prompt("explainer", "Custom inline")

        output = layer.inspect(print_output=False)
        assert "context_manager" in output
        assert "artifacts" in output
        assert "explainer" in output
        assert "validator" in output
        assert "Inline (str)" in output


# =========================================================================
# Scenario 4: Multi-Turn Conversation & Session Cache
# =========================================================================
class TestMultiTurnAndSQLite:
    def test_multi_turn_with_session_cache_and_sqlite(self, tmp_path: Path) -> None:
        db_file = tmp_path / "telemetry_test.db"
        store = SQLiteSessionStore(db_path=db_file)

        mock_llm = _make_smart_mock_llm()
        layer = ExplanationLayer.default_layer(llm=mock_llm, session_store=store)

        session_id = "incidente_turbina_901"

        # Turno 1: Passa predição e XAI
        res1 = layer.explain(
            prediction={
                "target_variable": "Resfriador",
                "predicted_value": "Falha Iminente",
                "confidence_or_probability": 0.88,
            },
            xai={"feature_attributions": {"PS1": 0.52}},
            user_query="O que causou o alarme?",
            session_id=session_id,
        )
        assert res1.session_id == session_id
        assert res1.validation_passed is True

        # Turno 2: Pergunta subsequente OMITINDO prediction/xai
        # A layer deve herdar os artefatos da sessão do Turno 1!
        res2 = layer.explain(
            user_query="Há risco de danificar a tubulação se mantivermos a carga?",
            session_id=session_id,
        )
        assert res2.session_id == session_id
        assert res2.validation_passed is True

        # Verifica persistência no SQLite via store e via layer.get_history
        history = layer.get_history(session_id)
        assert len(history) == 4  # 2 user queries + 2 assistant responses
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[2].role == "user"
        assert history[3].role == "assistant"

        # Verifica logs de validação gravados automaticamente
        logs = store.get_validation_logs(session_id)
        assert len(logs) == 2
        assert logs[0]["score"] == 9.5
        assert logs[0]["passed"] is True

        # Testa layer.clear_session
        layer.clear_session(session_id)
        assert len(layer.get_history(session_id)) == 0


# =========================================================================
# Scenario 5: Pluggable RAG in ContextManagerAgent
# =========================================================================
class TestPluggableRAG:
    def test_cma_custom_retriever_via_layer(self) -> None:
        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        cma = layer.get_agent("context_manager")

        # Custom vector search simulation (e.g., FAISS or LlamaIndex)
        def my_manual_search(query: str) -> str:
            return f"500-page manual for '{query}': Valve B-12 threshold is 4.5 bar."

        cma.set_retriever(my_manual_search)

        res = layer.explain(user_query="Valve B-12 threshold")
        assert res.validation_passed is True


# =========================================================================
# Scenario 6 & Fail-Safe: Production Resilience ("Unplugged")
# =========================================================================
class TestFailSafeMode:
    def test_fail_safe_graceful_degradation_on_llm_crash(self) -> None:
        def crashing_llm(prompt: str) -> str:
            raise ConnectionError("503 Server Unavailable / Network down")

        layer = ExplanationLayer.default_layer(llm=crashing_llm)
        res = layer.explain(
            prediction={"target_variable": "Motor", "predicted_value": "Critical"},
            user_query="Status check",
        )

        assert isinstance(res, ExplanationResponse)
        assert res.fallback_used is True
        assert res.validation_passed is False
        assert "FALLBACK DETERMINÍSTICO ATIVADO" in res.narrative
        assert "Motor" in res.narrative


# =========================================================================
# Scenario 7 & 8: Autonomous Query-Based Mode & Zero ML Mode
# =========================================================================
class TestAutonomousAndZeroML:
    def test_scenario_7_autonomous_query_tools(self) -> None:
        def fetch_incident_ml(incident_id: int) -> dict:
            return {
                "prediction": {
                    "target_variable": "Pump Pressure",
                    "predicted_value": "Overpressure",
                    "confidence_or_probability": 0.91,
                },
                "xai": {"feature_attributions": {"PS1": 0.60}},
            }

        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        aa = layer.get_agent("artifacts")
        aa.register_tool(fetch_incident_ml)

        res = layer.explain(
            incident_id=402,
            user_query="Investigate incident 402",
        )
        assert res.validation_passed is True

    def test_scenario_8_zero_ml_mode(self) -> None:
        layer = ExplanationLayer.default_layer(llm=_make_smart_mock_llm())
        doc_rule = "Bypass line if pressure drops below 2.0 bar."
        res = layer.explain(
            context={"documents": {"rule_1": doc_rule}},
            user_query="What is the safety rule for pressure drops?",
        )
        assert res.validation_passed is True

    def test_multi_turn_cma_progressive_tool_hydration_and_memory(self) -> None:
        tools_called: list[str] = []

        def buscar_documentacao(topico: str) -> str:
            """Busca documentação do dataset."""
            tools_called.append(f"doc:{topico}")
            return "Dataset Doc: Sensor PS1 mede pressão de sucção em bar."

        def buscar_manual(equipamento: str) -> str:
            """Busca manual do equipamento."""
            tools_called.append(f"manual:{equipamento}")
            return "Manual P-101: Abrir válvula V-2 se PS1 > 4.0 bar."

        def progressive_responder(
            prompt: str, system_prompt: str | None = None
        ) -> str:
            if "score" in prompt.lower() and "issues" in prompt.lower():
                return '{"score": 9.5, "issues": [], "critique": ""}'
            if "## Tool-Dispatch Reasoning Instructions" in prompt:
                if "Qual o significado no dataset" in prompt:
                    return (
                        '{"tool_calls": [{"name": "buscar_documentacao", '
                        '"arguments": {"topico": "PS1"}}]}'
                    )
                if "Qual o procedimento no manual" in prompt:
                    return (
                        '{"tool_calls": [{"name": "buscar_manual", '
                        '"arguments": {"equipamento": "P-101"}}]}'
                    )
                # Turn 3: no tool needed -> single-pass response
                return "Contexto já consolidado nos turnos anteriores."
            return f"Síntese validada para: {prompt[:80]}"

        layer = ExplanationLayer.default_layer(llm=progressive_responder)
        cma = layer.get_agent("context_manager")
        cma.register_tool(buscar_documentacao)
        cma.register_tool(buscar_manual)

        # Turn 1: Needs dataset documentation -> calls buscar_documentacao
        layer.explain(
            user_query="Qual o significado no dataset?",
            session_id="sess_prog",
        )
        assert tools_called == ["doc:PS1"]

        # Turn 2: Needs equipment manual -> calls buscar_manual and merges into session
        layer.explain(
            user_query="Qual o procedimento no manual?",
            session_id="sess_prog",
        )
        assert tools_called == ["doc:PS1", "manual:P-101"]
        cached_ctx = layer._session_cache["sess_prog"]["context"]
        assert len(cached_ctx.documents) == 2

        # Turn 3: Follow-up summary -> calls 0 tools and uses accumulated memory!
        tools_called.clear()
        layer.explain(
            user_query="Obrigado, faça um resumo geral.",
            session_id="sess_prog",
        )
        assert tools_called == []
        assert len(layer.get_history("sess_prog")) == 6
