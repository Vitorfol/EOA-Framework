"""Live integration tests for ExplanationLayer and default_layer with Gemini API (EP08).

Executed only when `GEMINI_API_KEY` is present in the environment or `.env` file.
Validates the developer-facing façade (`default_layer` and `ExplanationLayer.explain`)
against real Gemini model completions and LLM-as-a-Judge auditing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eoa import (
    ExplanationLayer,
    ExplanationResponse,
    SpecializedAgent,
    SQLiteSessionStore,
    default_layer,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="Skipping live integration test: GEMINI_API_KEY is not set.",
    ),
]


class TestLiveExplanationLayer:
    def test_live_default_layer_plug_and_play(self) -> None:
        """Validate 1-line `default_layer()` with real Gemini API (Zero-Config)."""
        layer = default_layer()

        response = layer.explain(
            prediction={
                "target_variable": "Condição do Resfriador Hidráulico",
                "predicted_value": "Eficiência Reduzida (Nível Crítico)",
                "confidence_or_probability": 0.88,
            },
            xai={
                "method": "SHAP",
                "feature_attributions": {"PS1": 0.55, "EPS1": 0.35},
            },
            context={
                "variables": {
                    "PS1": "Pressão de entrada (bar, normal <= 4.0 bar)",
                    "EPS1": "Potência da bomba principal (kW, normal <= 3.0 kW)",
                },
                "documents": {
                    "POP_04": (
                        "Quando PS1 sobe com sobrepotência em EPS1, "
                        "há obstrução ou cavitação no filtro de sucção."
                    )
                },
            },
            user_query="Explique o motivo deste alerta e qual procedimento adotar.",
        )

        print("\n" + "=" * 75)
        print("[LIVE EXPLANATION LAYER - PLUG & PLAY]")
        print(
            f"Validation Score: {response.validation_score}/10.0 "
            f"(Passed: {response.validation_passed}, "
            f"Attempts: {response.attempts_used})"
        )
        print(f"Narrative:\n{response.narrative}")
        print("=" * 75)

        assert isinstance(response, ExplanationResponse)
        assert len(response.narrative) > 30
        assert response.validation_score is not None
        assert 0.0 <= response.validation_score <= 10.0

    def test_live_multi_turn_session_and_specialist_agent(
        self, tmp_path: Path
    ) -> None:
        """Validate live multi-turn SQLite session with a SpecializedAgent."""
        db_file = tmp_path / "live_layer_audit.db"
        store = SQLiteSessionStore(db_path=db_file)

        layer = ExplanationLayer(session_store=store)
        layer.create_default_agents(threshold=7.0)

        # Register a SpecializedAgent
        layer.register_agent(
            SpecializedAgent(
                name="safety_engineer",
                role="Engenheiro de Segurança Operacional",
                prompt=(
                    "Você é um engenheiro de segurança industrial. "
                    "Aponte o cuidado imediato de segurança para: {user_query}"
                ),
            )
        )

        session_id = "live_incident_909"

        # Turn 1: Pass artifacts
        res1 = layer.explain(
            prediction={
                "target_variable": "Válvula Hidráulica Principal",
                "predicted_value": "Atraso de Fechamento Severo",
                "confidence_or_probability": 0.91,
            },
            xai={"feature_attributions": {"FS1_Fluxo": 0.62, "PS2_Pressao": 0.38}},
            user_query="Qual é o diagnóstico inicial da válvula?",
            session_id=session_id,
        )
        assert res1.narrative
        assert res1.session_id == session_id

        # Turn 2: Follow-up omitting prediction/xai (uses session artifact cache)
        res2 = layer.explain(
            user_query="Qual medida de segurança devemos tomar antes da manutenção?",
            session_id=session_id,
        )
        assert res2.narrative
        assert len(store.get_history(session_id)) == 4
        assert len(store.get_validation_logs(session_id)) == 2
