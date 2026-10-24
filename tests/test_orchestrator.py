"""Deterministic unit and component integration tests for OrchestratorEngine (EP07) using MockLLM.

Covers:
1. Happy Path (Push Mode, Sync & Async parity, parallel execution).
2. Pull Mode (Query-based autonomous hydration from a real SQLite incident database).
3. Zero ML Mode (Progressive Ramp conceptual queries with 0 AA LLM tokens).
4. Progressive Mixed Session (Zero ML -> Pull -> Push within the same SQLite session).
5. Stateless vs Persistent Session Management (auto-generated UUID vs explicit ID).
6. Refinement Loop & Quality Gate (Self-correction, cutoff exhaustion, parse_error).
7. Unplugged Fail-Safe Mode (Provider-agnostic exception resilience).
8. Modular Agent Composition ("Developer is King": optional CMA, AA, VA, or SAs).
9. Structured Observability Logging (capturing eoa.orchestrator logs).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from eoa import (
    ArtifactsAgent,
    CallableLLMAdapter,
    ContextManagerAgent,
    DomainContext,
    ExplainerAgent,
    ExplanationResponse,
    InMemorySessionStore,
    OrchestratorEngine,
    PredictionArtifact,
    SpecializedAgent,
    SQLiteSessionStore,
    ValidatorAgent,
    XAIArtifact,
)
from tests.mocks.mock_llm import MockLLM


# =========================================================================
# Helper: Sequence LLM that responds based on prompt content or queue
# =========================================================================
def _make_smart_mock_llm(
    va_responses: list[str] | None = None,
    ea_responses: list[str] | None = None,
) -> MockLLM:
    """Create a MockLLM with deterministic routing for CMA, AA, EA, VA, and SAs."""
    va_queue = list(
        va_responses or ['{"score": 9.2, "issues": [], "critique": ""}']
    )
    ea_queue = list(
        ea_responses
        or [
            "O alerta foi causado pela queda de pressão no sensor PS1 (0.48), "
            "indicando risco de 85% de degradação no resfriador. Recomenda-se "
            "inspecionar a válvula de sucção imediatamente."
        ]
    )

    def responder(prompt: str, system_prompt: str | None = None) -> str:
        if "You are the Context Manager" in prompt:
            return "Key Variables:\n- PS1: Pressão de entrada (limite 4.0 bar)."
        if "You are the Artifacts Analyst" in prompt:
            return "Outcome: Risco 85%. Dominant Drivers: PS1 (+0.48)."
        if "You are the Validator" in prompt:
            return va_queue.pop(0) if len(va_queue) > 1 else va_queue[0]
        if "You are the Explainer" in prompt:
            return ea_queue.pop(0) if len(ea_queue) > 1 else ea_queue[0]
        return "Parecer técnico do especialista concluído."

    return MockLLM(custom_generator=responder)


# =========================================================================
# 1. Happy Path: Push Mode (Sync & Async Parity)
# =========================================================================
class TestOrchestratorPushMode:
    def test_push_mode_sync_and_async_execution(self) -> None:
        llm = _make_smart_mock_llm()
        engine = OrchestratorEngine(
            llm=llm,
            cma=ContextManagerAgent(),
            aa=ArtifactsAgent(),
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
        )

        pred = PredictionArtifact(
            target_variable="Cooler Condition",
            predicted_value="Degraded",
            confidence_or_probability=0.85,
        )
        xai = XAIArtifact(
            method="SHAP",
            feature_attributions={"PS1": 0.48, "EPS1": 0.31},
        )
        ctx = DomainContext(
            variables={"PS1": "Inlet Pressure", "EPS1": "Motor Power"},
            thresholds={"PS1_max": "4.0 bar"},
        )

        # Synchronous run
        res_sync = engine.run(
            user_query="Por que o resfriador entrou em estado degradado?",
            prediction=pred,
            xai=xai,
            context=ctx,
        )
        assert isinstance(res_sync, ExplanationResponse)
        assert res_sync.fallback_used is False
        assert res_sync.validation_passed is True
        assert res_sync.validation_score == 9.2
        assert res_sync.attempts_used == 1
        assert res_sync.session_id is not None and len(res_sync.session_id) > 0
        assert "PS1" in res_sync.narrative

        trace_names = [t.agent_name for t in res_sync.execution_trace]
        assert trace_names == [
            "context_manager",
            "artifacts",
            "explainer",
            "validator",
        ]

        # Asynchronous run
        res_async = asyncio.run(
            engine.arun(
                user_query="Por que o resfriador entrou em estado degradado?",
                prediction=pred,
                xai=xai,
                context=ctx,
            )
        )
        assert res_async.fallback_used is False
        assert res_async.validation_passed is True
        assert res_async.validation_score == 9.2
        assert res_async.attempts_used == 1


# =========================================================================
# 2. Pull Mode: Autonomous Query Hydration with Real SQLite Incident DB
# =========================================================================
class TestOrchestratorPullMode:
    def test_pull_mode_hydrates_from_sqlite_incident_database(
        self, tmp_path: Path
    ) -> None:
        # 1. Setup a simulated industrial SCADA / MLflow SQLite database
        scada_db = tmp_path / "industrial_scada.db"
        with sqlite3.connect(scada_db) as conn:
            conn.execute(
                """
                CREATE TABLE incidents (
                    incident_id INTEGER PRIMARY KEY,
                    target_variable TEXT,
                    predicted_value TEXT,
                    confidence REAL,
                    ps1_shap REAL,
                    eps1_shap REAL,
                    manual_rule TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO incidents VALUES (
                    402,
                    'Hydraulic Pump Cavitation',
                    'Critical Alert',
                    0.91,
                    0.55,
                    0.35,
                    'Section 7.1: Inspect suction valve when PS1 attribution > 0.50'
                )
                """
            )
            conn.commit()

        # 2. Define tools querying the SQLite database by incident_id
        def fetch_ml_artifacts_from_db(incident_id: int) -> dict[str, Any]:
            with sqlite3.connect(scada_db) as conn:
                sql = (
                    "SELECT target_variable, predicted_value, confidence, "
                    "ps1_shap, eps1_shap FROM incidents WHERE incident_id = ?"
                )
                row = conn.execute(sql, (incident_id,)).fetchone()
            return {
                "prediction": {
                    "target_variable": row[0],
                    "predicted_value": row[1],
                    "confidence_or_probability": row[2],
                },
                "xai": {
                    "method": "SHAP",
                    "feature_attributions": {"PS1": row[3], "EPS1": row[4]},
                },
            }

        def fetch_domain_manual_from_db(incident_id: int) -> dict[str, Any]:
            with sqlite3.connect(scada_db) as conn:
                row = conn.execute(
                    "SELECT manual_rule FROM incidents WHERE incident_id = ?",
                    (incident_id,),
                ).fetchone()
            return {
                "variables": {"PS1": "Suction Line Pressure"},
                "documents": {"POP_402": row[0]},
            }

        cma = ContextManagerAgent(tools=[fetch_domain_manual_from_db])
        aa = ArtifactsAgent(tools=[fetch_ml_artifacts_from_db])
        ea = ExplainerAgent()
        va = ValidatorAgent(threshold=7.0)

        llm = _make_smart_mock_llm()
        engine = OrchestratorEngine(llm=llm, cma=cma, aa=aa, ea=ea, va=va)

        # 3. Call orchestrator passing ONLY incident_id=402 (Pull Mode)
        response = engine.run(
            user_query="Investigue o incidente 402 e recomende ação.",
            incident_id=402,
        )

        assert response.fallback_used is False
        assert response.validation_passed is True
        # Verify that prompts sent to LLM contained the SQLite-hydrated data
        prompts_joined = "\n".join(c.prompt for c in llm.calls)
        assert "Hydraulic Pump Cavitation" in prompts_joined
        assert "Section 7.1: Inspect suction valve" in prompts_joined
        assert "+0.5500" in prompts_joined


# =========================================================================
# 3. Zero ML Mode: Conceptual Onboarding with 0 AA LLM Tokens
# =========================================================================
class TestOrchestratorZeroMLMode:
    def test_zero_ml_skips_artifacts_llm_call(self) -> None:
        llm = _make_smart_mock_llm()
        engine = OrchestratorEngine(
            llm=llm,
            cma=ContextManagerAgent(),
            aa=ArtifactsAgent(),
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
        )

        response = engine.run(
            user_query="Como funciona o circuito de alívio hidráulico?",
            context={
                "documents": {
                    "manual_cap1": "A válvula abre automaticamente acima de 4.0 bar."
                }
            },
        )

        assert response.fallback_used is False
        assert response.validation_passed is True

        aa_trace = next(
            t for t in response.execution_trace if t.agent_name == "artifacts"
        )
        assert "Zero ML mode" in aa_trace.summary
        # Total LLM calls: CMA (1) + EA (1) + VA (1) = 3 (AA made 0 LLM calls!)
        assert llm.call_count == 3


# =========================================================================
# 4. Progressive Mixed Session: Zero ML -> Pull -> Push in Same Session
# =========================================================================
class TestProgressiveMixedSession:
    def test_single_session_transitions_zero_ml_then_pull_then_push(
        self, tmp_path: Path
    ) -> None:
        session_db = tmp_path / "operator_sessions.db"
        store = SQLiteSessionStore(session_db)

        def fetch_incident_tool(incident_id: int) -> dict[str, Any]:
            return {
                "prediction": {
                    "target_variable": "Valve State",
                    "predicted_value": "Partial Blockage",
                    "confidence_or_probability": 0.78,
                },
                "xai": {"feature_attributions": {"FS1_flow": -0.62}},
            }

        engine = OrchestratorEngine(
            llm=_make_smart_mock_llm(),
            cma=ContextManagerAgent(),
            aa=ArtifactsAgent(tools=[fetch_incident_tool]),
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
            session_store=store,
        )

        session_id = "turno_operador_hidreletrica_01"

        # TURN 1: Zero ML (Conceptual onboarding question)
        res_turn1 = engine.run(
            user_query="Turno 1: Qual é a vazão nominal do sensor FS1_flow?",
            context={
                "variables": {"FS1_flow": "Vazão volumétrica (normal: 10 L/min)"}
            },
            session_id=session_id,
        )
        assert res_turn1.session_id == session_id
        assert len(store.get_history(session_id)) == 2  # 1 user + 1 assistant

        # TURN 2: Pull Mode (Operator triggers investigation by incident_id)
        res_turn2 = engine.run(
            user_query="Turno 2: O alarme tocou. Analise o incidente 88.",
            incident_id=88,
            session_id=session_id,
        )
        assert res_turn2.session_id == session_id
        assert len(store.get_history(session_id)) == 4  # 2 turns stored

        # TURN 3: Push Mode (Operator passes a new prediction directly)
        res_turn3 = engine.run(
            user_query="Turno 3: Com a limpeza do filtro, o risco caiu?",
            prediction={
                "target_variable": "Valve State",
                "predicted_value": "Optimal",
                "confidence_or_probability": 0.96,
            },
            xai={"feature_attributions": {"FS1_flow": 0.10}},
            session_id=session_id,
        )
        assert res_turn3.session_id == session_id
        history = store.get_history(session_id)
        assert len(history) == 6  # All 3 turns (6 messages) persisted in SQLite!

        # Verify Turn 3 Explainer prompt received dialogue history from Turns 1 & 2
        ea_prompts = [
            c.prompt
            for c in engine.llm.calls  # type: ignore[attr-defined]
            if "You are the Explainer" in c.prompt
        ]
        last_ea_prompt = ea_prompts[-1]
        assert "Turno 1: Qual é a vazão nominal" in last_ea_prompt
        assert "Turno 2: O alarme tocou" in last_ea_prompt


# =========================================================================
# 5. Stateless vs Persistent Session Management
# =========================================================================
class TestSessionManagement:
    def test_omitting_session_id_runs_statelessly_without_polluting_store(
        self,
    ) -> None:
        store = InMemorySessionStore()
        engine = OrchestratorEngine(
            llm=_make_smart_mock_llm(),
            ea=ExplainerAgent(),
            session_store=store,
        )

        # Call without session_id
        res = engine.run(user_query="Pergunta rápida descartável")
        assert res.session_id is not None
        assert len(res.session_id) == 32  # Generated UUID4 hex
        assert store.list_sessions() == []  # Store remains clean!


# =========================================================================
# 6. Refinement Loop & Quality Gate (Self-Correction & Cutoff)
# =========================================================================
class TestRefinementLoopAndQualityGate:
    def test_refinement_loop_recovers_on_second_attempt(self) -> None:
        critique_json = (
            '{"score": 5.0, "issues": ["Faltou o limiar"], '
            '"critique": "Inclua o limiar de 4.0 bar."}'
        )
        llm = _make_smart_mock_llm(
            ea_responses=[
                "Rascunho 1 incompleto sem citar o limiar de 4.0 bar.",
                "Rascunho 2 corrigido: O sensor PS1 ultrapassou o limiar de 4.0 bar.",
            ],
            va_responses=[
                critique_json,
                '{"score": 9.0, "issues": [], "critique": ""}',
            ],
        )
        engine = OrchestratorEngine(
            llm=llm,
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
            max_attempts=3,
        )

        res = engine.run(user_query="Explique o alerta.")
        assert res.attempts_used == 2
        assert res.validation_passed is True
        assert res.validation_score == 9.0
        assert res.fallback_used is False
        assert "Rascunho 2 corrigido" in res.narrative

        # Check that second EA call received the critique from first VA call
        ea_prompts = [
            c.prompt for c in llm.calls if "You are the Explainer" in c.prompt
        ]
        assert len(ea_prompts) == 2
        assert "Inclua o limiar de 4.0 bar." in ea_prompts[1]

    def test_refinement_loop_exhaustion_sets_fallback_warning(self) -> None:
        v1 = '{"score": 2.0, "issues": ["Alucinações"], "critique": "Corrija."}'
        v2 = '{"score": 6.5, "issues": ["Falta detalhe"], "critique": "Ajuste."}'
        v3 = '{"score": 4.0, "issues": ["Piorou"], "critique": "Tente de novo."}'

        llm = _make_smart_mock_llm(
            ea_responses=[
                "Rascunho 1 com nota 2.0 (muito ruim)",
                "Rascunho 2 com nota 6.5 (melhor tentativa obtida)",
                "Rascunho 3 com nota 4.0 (tentativa final regrediu)",
            ],
            va_responses=[v1, v2, v3],
        )
        engine = OrchestratorEngine(
            llm=llm,
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.5),
            max_attempts=3,
        )

        res = engine.run(user_query="Explique o alerta.")
        assert res.attempts_used == 3
        assert res.validation_passed is False
        assert res.validation_score == 4.0  # Last score
        assert res.fallback_used is True
        # Guarantees operator gets the BEST draft rather than worst regression
        assert "Rascunho 2 com nota 6.5" in res.narrative
        assert res.execution_trace[-1].status == "fallback"

    def test_validator_parse_error_triggers_deterministic_fallback_immediately(
        self,
    ) -> None:
        llm = _make_smart_mock_llm(
            va_responses=["Resposta corrompida sem JSON nem nota."]
        )
        engine = OrchestratorEngine(
            llm=llm,
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
            max_attempts=3,
        )

        res = engine.run(
            user_query="Explique o alerta.",
            prediction={"target_variable": "Pump", "predicted_value": "Fail"},
        )
        assert res.attempts_used == 1  # Does not waste retries on broken judge format
        assert res.fallback_used is True
        assert res.validation_passed is False
        assert "[FALLBACK DETERMINÍSTICO ATIVADO" in res.narrative
        assert "Pump" in res.narrative


# =========================================================================
# 7. Unplugged Fail-Safe Mode (Provider-Agnostic Exception Resilience)
# =========================================================================
class TestUnpluggedFailSafe:
    def test_failsafe_on_broken_callable_llm_simulating_openai_or_local_ollama(
        self,
    ) -> None:
        """Verify custom callables (OpenAI, Ollama, vLLM) crashing trigger fallback."""

        def broken_local_ollama_or_openai(prompt: str) -> str:
            raise ConnectionError(
                "Connection refused: Local Ollama / OpenAI gateway unreachable"
            )

        custom_llm = CallableLLMAdapter(broken_local_ollama_or_openai)
        engine = OrchestratorEngine(
            llm=custom_llm,
            cma=ContextManagerAgent(),
            aa=ArtifactsAgent(),
            ea=ExplainerAgent(),
            va=ValidatorAgent(),
        )

        res = engine.run(
            user_query="Qual o diagnóstico da turbina?",
            prediction={
                "target_variable": "Turbine Bearing",
                "predicted_value": "Overheating",
                "confidence_or_probability": 0.94,
            },
            xai={"feature_attributions": {"TS1_temp": 0.81}},
            context={"variables": {"TS1_temp": "Bearing Temperature (max 85C)"}},
        )

        # Host application NEVER crashes!
        assert isinstance(res, ExplanationResponse)
        assert res.fallback_used is True
        assert res.validation_passed is False
        assert "[FALLBACK DETERMINÍSTICO ATIVADO" in res.narrative
        assert "Turbine Bearing" in res.narrative
        assert "Overheating" in res.narrative
        assert "94.00%" in res.narrative
        assert "TS1_temp: +0.8100" in res.narrative
        assert res.execution_trace[-1].status == "fallback"


# =========================================================================
# 8. Modular Composition ("Developer is King") & Logging Observability
# =========================================================================
class TestModularCompositionAndLogging:
    def test_orchestrator_with_specialized_agent_and_no_validator(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        llm = _make_smart_mock_llm()
        sa = SpecializedAgent(
            name="cavitation_specialist",
            prompt="Analyze cavitation for {user_query}",
        )
        engine = OrchestratorEngine(
            llm=llm,
            aa=ArtifactsAgent(),
            ea=ExplainerAgent(),
            specialized_agents=[sa],
            va=None,  # Developer opted out of ValidatorAgent
        )

        with caplog.at_level(logging.INFO, logger="eoa.orchestrator"):
            res = engine.run(
                user_query="Check cavitation",
                prediction={"target_variable": "Pump", "predicted_value": "Alert"},
            )

        assert res.fallback_used is False
        assert res.validation_passed is None  # Validator was disabled
        assert res.validation_score is None
        assert res.attempts_used == 1
        trace_names = [t.agent_name for t in res.execution_trace]
        assert "cavitation_specialist" in trace_names
        assert "validator" not in trace_names

        # Verify structured logs were emitted to eoa.orchestrator
        log_text = "\n".join(caplog.messages)
        assert "Starting synchronous orchestration pipeline" in log_text
        assert "Executing Phase 1 evidence agents concurrently" in log_text
        assert "Orchestration completed in" in log_text
