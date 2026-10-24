"""Live end-to-end integration tests for OrchestratorEngine with Gemini API.

Executed only when `GEMINI_API_KEY` is present in the environment.
Includes visual console prints (`pytest -s`) and logger instrumentation so the
developer can inspect the real multi-turn conversation, SQLite hydration, and
LLM-as-a-Judge quality gate in action.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from eoa import (
    ArtifactsAgent,
    ContextManagerAgent,
    ExplainerAgent,
    GeminiProvider,
    OrchestratorEngine,
    SQLiteSessionStore,
    ValidatorAgent,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="Skipping live integration test: GEMINI_API_KEY is not set.",
    ),
]


@pytest.fixture
def live_gemini() -> GeminiProvider:
    model_name = os.environ.get("GEMINI_MODEL") or GeminiProvider.DEFAULT_MODEL
    return GeminiProvider(model=model_name)


class TestLiveOrchestratorEndToEnd:
    def test_live_progressive_session_zero_ml_to_pull_to_push(
        self,
        live_gemini: GeminiProvider,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full live workflow transitioning Zero ML -> Pull (SQLite tool) -> Push."""
        caplog.set_level(logging.INFO, logger="eoa.orchestrator")

        # 1. Setup SQLite Incident DB for Pull Mode and SQLiteSessionStore for Memory
        scada_db = tmp_path / "live_scada.db"
        with sqlite3.connect(scada_db) as conn:
            conn.execute(
                "CREATE TABLE alerts ("
                "id INT PRIMARY KEY, target TEXT, val TEXT, "
                "conf REAL, ps1 REAL, eps1 REAL)"
            )
            conn.execute(
                "INSERT INTO alerts VALUES "
                "(402, 'Cooler Efficiency', 'Degraded', 0.89, 0.54, 0.36)"
            )
            conn.commit()

        def query_alert_db(incident_id: int) -> dict[str, Any]:
            with sqlite3.connect(scada_db) as conn:
                r = conn.execute(
                    "SELECT target, val, conf, ps1, eps1 FROM alerts WHERE id = ?",
                    (incident_id,),
                ).fetchone()
            return {
                "prediction": {
                    "target_variable": r[0],
                    "predicted_value": r[1],
                    "confidence_or_probability": r[2],
                },
                "xai": {
                    "method": "SHAP",
                    "feature_attributions": {"PS1": r[3], "EPS1": r[4]},
                },
            }

        session_store = SQLiteSessionStore(tmp_path / "live_sessions.db")
        engine = OrchestratorEngine(
            llm=live_gemini,
            cma=ContextManagerAgent(),
            aa=ArtifactsAgent(tools=[query_alert_db]),
            ea=ExplainerAgent(),
            va=ValidatorAgent(threshold=7.0),
            session_store=session_store,
            max_attempts=2,
        )

        session_id = "live_session_turbine_01"

        # ------------------------------------------------------------------
        # TURN 1: Zero ML Mode (Conceptual question with domain manual)
        # ------------------------------------------------------------------
        res1 = engine.run(
            user_query="Qual é a função do sensor PS1 e qual seu limite operacional?",
            context={
                "variables": {
                    "PS1": "Pressão de entrada do circuito de refrigeração (bar)",
                    "EPS1": "Potência elétrica da bomba principal (kW)",
                },
                "thresholds": {"PS1_max": "4.0 bar", "EPS1_max": "3.0 kW"},
            },
            session_id=session_id,
        )
        print("\n" + "=" * 75)
        print("[LIVE TURN 1 - ZERO ML MODE]")
        print(f"Session ID: {res1.session_id}")
        print(
            f"Validation Score: {res1.validation_score} "
            f"(Passed: {res1.validation_passed})"
        )
        print(f"Narrative:\n{res1.narrative}")
        print("=" * 75)

        assert res1.narrative
        assert res1.session_id == session_id

        # ------------------------------------------------------------------
        # TURN 2: Pull Mode (Autonomous SQLite Tool Hydration via incident_id)
        # ------------------------------------------------------------------
        res2 = engine.run(
            user_query=(
                "O alarme 402 disparou agora. O que está acontecendo e o que fazer?"
            ),
            incident_id=402,
            context={
                "variables": {
                    "PS1": "Pressão de entrada (bar)",
                    "EPS1": "Potência (kW)",
                },
                "documents": {
                    "POP_04": (
                        "Em caso de degradação por PS1, inspecionar válvula de alívio."
                    )
                },
            },
            session_id=session_id,
        )
        print("\n" + "=" * 75)
        print("[LIVE TURN 2 - PULL MODE (SQLite Tool Hydration)]")
        print(
            f"Validation Score: {res2.validation_score} "
            f"(Passed: {res2.validation_passed}, Attempts: {res2.attempts_used})"
        )
        print(f"Narrative:\n{res2.narrative}")
        print("=" * 75)

        assert res2.narrative
        assert len(session_store.get_history(session_id)) == 4

        # ------------------------------------------------------------------
        # TURN 3: Push Mode (Direct Prediction After Operator Intervention)
        # ------------------------------------------------------------------
        res3 = engine.run(
            user_query=(
                "A equipe abriu a válvula de alívio e fizemos nova leitura. "
                "A situação normalizou?"
            ),
            prediction={
                "target_variable": "Cooler Efficiency",
                "predicted_value": "Normal Operation",
                "confidence_or_probability": 0.95,
            },
            xai={
                "method": "SHAP",
                "feature_attributions": {"PS1": 0.05, "EPS1": 0.04},
            },
            session_id=session_id,
        )
        print("\n" + "=" * 75)
        print("[LIVE TURN 3 - PUSH MODE (Post-Intervention)]")
        print(
            f"Validation Score: {res3.validation_score} "
            f"(Passed: {res3.validation_passed})"
        )
        print(f"Narrative:\n{res3.narrative}")
        print("\n[ORCHESTRATOR LOGS CAPTURED]:")
        for msg in caplog.messages:
            print(f"  -> {msg}")
        print("=" * 75)

        assert res3.narrative
        history = session_store.get_history(session_id)
        assert len(history) == 6
