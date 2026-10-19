"""Live integration tests for EOA agents against real Gemini LLM API.

These tests are only executed when `GEMINI_API_KEY` is present in the environment.
They validate that each agent (CMA, AA, SA, EA, VA) and the end-to-end chained
pipeline work seamlessly with real Gemini completions and JSON formatting.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from eoa.agents import (
    AgentState,
    ArtifactsAgent,
    ContextManagerAgent,
    ExplainerAgent,
    SpecializedAgent,
    ValidatorAgent,
)
from eoa.contracts import (
    DomainContext,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.llm.gemini import GeminiProvider
from eoa.memory.base import ChatMessage

# Skip entire module if GEMINI_API_KEY is missing
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="Skipping live integration test: GEMINI_API_KEY is not set.",
    ),
]


@pytest.fixture
def live_gemini_llm() -> GeminiProvider:
    """Fixture providing a configured GeminiProvider using environment credentials."""
    model_name = os.environ.get("GEMINI_MODEL") or GeminiProvider.DEFAULT_MODEL
    return GeminiProvider(model=model_name)


class TestLiveAgentsWithGemini:
    """Live validation suite for each individual agent and combined pipeline."""

    def test_live_cma_context_manager(self, live_gemini_llm: GeminiProvider) -> None:
        """Validate ContextManagerAgent with real Gemini LLM."""
        cma = ContextManagerAgent()
        context = DomainContext(
            variables={
                "PS1": "Pressão de entrada da linha de fluido de refrigeração",
                "EPS1": "Potência do motor da bomba principal (kW)",
                "TS2": "Temperatura no trocador de calor (°C)",
            },
            thresholds={
                "PS1_max": 4.5,
                "EPS1_nominal": 2.8,
            },
            documents={
                "manual_refrig": (
                    "Seção 4.2: Quando EPS1 ultrapassa 3.0 kW com PS1 abaixo de 3.0"
                    " bar, há indicativo de cavitação no filtro de sucção."
                )
            },
        )
        state = AgentState(
            user_query="Por que a bomba parece estar com comportamento anômalo?",
            context=context,
        )

        result = cma.run(state, live_gemini_llm)

        assert result.output
        assert len(result.output) > 20
        assert state.context_summary == result.output
        assert result.trace.agent_name == "context_manager"
        assert result.trace.execution_time_ms > 0
        assert result.trace.status == "success"

    def test_live_aa_artifacts_agent(self, live_gemini_llm: GeminiProvider) -> None:
        """Validate ArtifactsAgent interpreting prediction and XAI with real Gemini."""
        aa = ArtifactsAgent()
        pred = PredictionArtifact(
            model_name="CoolerFailurePredictor_v2",
            target_variable="Condição do Resfriador",
            predicted_value=1,
            confidence_or_probability=0.84,
            status_label="Degradação Severa / Risco Iminente",
        )
        xai = XAIArtifact(
            method="shap",
            feature_attributions={
                "PS1": 0.48,
                "EPS1": 0.36,
                "TS2": 0.16,
            },
        )
        state = AgentState(
            user_query="Analise a predição e as causas identificadas.",
            prediction=pred,
            xai=xai,
        )

        result = aa.run(state, live_gemini_llm)

        assert result.output
        assert len(result.output) > 20
        assert state.artifacts_summary == result.output
        assert result.trace.agent_name == "artifacts"
        assert result.trace.execution_time_ms > 0
        assert result.trace.status == "success"

    def test_live_specialized_agent(self, live_gemini_llm: GeminiProvider) -> None:
        """Validate SpecializedAgent executing custom domain persona."""
        turbine_expert = SpecializedAgent(
            name="turbine_hydraulic_expert",
            role="Especialista em Sistemas Hidráulicos de Turbinas",
            prompt=(
                "Você é um engenheiro sênior especialista em turbinas. "
                "Forneça recomendações práticas de mitigação. "
                "Pergunta: {user_query}\nContexto: {context_summary}"
            ),
        )
        state = AgentState(
            user_query="Qual medida emergencial de campo deve ser adotada?",
            context_summary=(
                "Bomba hidráulica apresentando cavitação com sobrepotência elétrica."
            ),
        )

        result = turbine_expert.run(state, live_gemini_llm)

        assert result.output
        assert len(result.output) > 20
        assert "turbine_hydraulic_expert" in state.specialist_insights
        assert state.specialist_insights["turbine_hydraulic_expert"] == result.output
        assert result.trace.execution_time_ms > 0

    def test_live_explainer_agent(self, live_gemini_llm: GeminiProvider) -> None:
        """Validate ExplainerAgent synthesizing full narrative with real Gemini."""
        ea = ExplainerAgent()
        history = [
            ChatMessage(role="user", content="Olá, o alarme 402 acabou de soar."),
            ChatMessage(
                role="assistant",
                content="Estou coletando a telemetria do sistema de refrigeração.",
            ),
        ]
        state = AgentState(
            user_query="Explique de forma clara o que está acontecendo e o risco.",
            context_summary=(
                "Circuito de Resfriamento. Manual recomenda bypass se PS1 cair."
            ),
            artifacts_summary=(
                "Predição: Degradação Severa (84% risco). Fator principal: PS1."
            ),
            chat_history=history,
        )

        result = ea.run(state, live_gemini_llm)

        assert result.output
        assert len(result.output) > 30
        assert state.draft_explanation == result.output
        assert result.trace.agent_name == "explainer"
        assert result.trace.execution_time_ms > 0
        assert result.trace.status == "success"

    def test_live_validator_agent_llm_as_a_judge(
        self, live_gemini_llm: GeminiProvider
    ) -> None:
        """Validate ValidatorAgent auditing an explanation with real Gemini JSON."""
        va = ValidatorAgent(threshold=7.0)
        state = AgentState(
            user_query="Explique a falha detectada.",
            artifacts_summary=(
                "Predição indica risco de 84% no resfriador devido à pressão PS1."
            ),
            context_summary=(
                "Manual 4.2: pressão PS1 abaixo de 3.0 bar requer inspeção da válvula."
            ),
            draft_explanation=(
                "Detectamos um risco de 84% de falha no sistema de resfriamento. "
                "O sensor de pressão PS1 é o principal fator causador. De acordo com "
                "o manual 4.2, a equipe deve inspecionar a válvula de sucção."
            ),
        )

        result = va.run(state, live_gemini_llm)

        assert state.validation_result is not None
        assert 0.0 <= state.validation_result.score <= 10.0
        assert isinstance(state.validation_result.passed, bool)
        assert isinstance(state.validation_result.critique, str)
        assert result.trace.agent_name == "validator"
        assert result.trace.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_live_end_to_end_pipeline_async(
        self, live_gemini_llm: GeminiProvider
    ) -> None:
        """Validate complete multi-agent pipeline asynchronously end-to-end."""
        cma = ContextManagerAgent()
        aa = ArtifactsAgent()
        expert = SpecializedAgent(
            name="safety_compliance_agent",
            prompt=(
                "Você é um auditor de segurança industrial. "
                "Aponte a criticidade para a equipe: "
                "Query: {user_query}\nArtefatos: {artifacts_summary}\n"
                "Contexto: {context_summary}"
            ),
        )
        ea = ExplainerAgent()
        va = ValidatorAgent(threshold=7.0)

        # Build comprehensive real-world state
        state = AgentState(
            user_query="Qual é a gravidade desta anomalia e quais passos seguir?",
            context=DomainContext(
                variables={"PS1": "Pressão de linha", "EPS1": "Potência do motor"},
                documents={
                    "POP-01": "Em falha de refrigeração, reduzir carga para 50%."
                },
            ),
            prediction=PredictionArtifact(
                target_variable="Falha no Circuito de Refrigeração",
                predicted_value=1,
                confidence_or_probability=0.88,
                status_label="Alerta Crítico",
            ),
            xai=XAIArtifact(
                method="shap",
                feature_attributions={"PS1": 0.55, "EPS1": 0.45},
            ),
        )

        # 1. Parallel execution of CMA and AA (orchestration pattern)
        await asyncio.gather(
            cma.arun(state, live_gemini_llm),
            aa.arun(state, live_gemini_llm),
        )
        assert state.context_summary is not None
        assert state.artifacts_summary is not None

        # 2. Specialist analysis
        await expert.arun(state, live_gemini_llm)
        assert "safety_compliance_agent" in state.specialist_insights

        # 3. Explainer synthesis
        await ea.arun(state, live_gemini_llm)
        assert state.draft_explanation is not None
        assert len(state.draft_explanation) > 30

        # 4. Validator LLM-as-a-Judge audit
        await va.arun(state, live_gemini_llm)
        assert state.validation_result is not None
        assert 0.0 <= state.validation_result.score <= 10.0
        assert state.validation_result.critique is not None
