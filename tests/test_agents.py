"""Comprehensive unit tests for EOA agents (CMA, AA, EA, VA, Specialized) and Tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eoa.agents import (
    AgentResult,
    AgentState,
    ArtifactsAgent,
    ContextManagerAgent,
    ContextRetriever,
    ExplainerAgent,
    SpecializedAgent,
    StructuredContextRetriever,
    ValidationResult,
    ValidatorAgent,
)
from eoa.contracts import (
    DomainContext,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.memory.base import ChatMessage
from eoa.tools import Tool
from tests.mocks.mock_llm import MockLLM


# =========================================================================
# 1. Tests for Tool abstraction
# =========================================================================
class TestTool:
    def test_sync_tool_inspection_and_execution(self) -> None:
        def fetch_data(item_id: int, scale: float = 1.0) -> dict:
            """Fetch simulated item data."""
            return {"id": item_id, "scaled": item_id * scale}

        tool = Tool(fetch_data)
        assert tool.name == "fetch_data"
        assert "Fetch simulated item data." in tool.description
        assert tool.required_parameters == ["item_id"]
        assert tool.can_execute_with({"item_id": 10}) is True
        assert tool.can_execute_with({"other": 10}) is False

        res = tool.execute(item_id=5, scale=2.0)
        assert res == {"id": 5, "scaled": 10.0}

    def test_async_tool_execution(self) -> None:
        async def async_fetch(resource_id: str) -> str:
            await asyncio.sleep(0.001)
            return f"resource_{resource_id}"

        tool = Tool(async_fetch)
        assert tool.is_async is True
        assert tool.can_execute_with({"resource_id": "abc"}) is True

        res = tool.execute(resource_id="abc")
        assert res == "resource_abc"

        async def _run() -> None:
            ares = await tool.aexecute(resource_id="xyz")
            assert ares == "resource_xyz"

        asyncio.run(_run())

    def test_tool_callable_validation(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            Tool("not_a_callable")  # type: ignore[arg-type]


# =========================================================================
# 2. Tests for ContextManagerAgent (CMA) and ContextRetriever
# =========================================================================
class TestContextManagerAgent:
    def test_structured_retriever_formatting(self) -> None:
        ctx = DomainContext(
            variables={"PS1": "Inlet Pressure", "EPS1": "Motor Power"},
            thresholds={"PS1": "4.0 bar"},
            documents={"manual": "Chapter 4: Hydraulic circuit."},
        )
        retriever = StructuredContextRetriever()
        text = retriever.retrieve("What is PS1?", ctx)

        assert "Inlet Pressure" in text
        assert "4.0 bar" in text
        assert "Chapter 4: Hydraulic circuit." in text

    def test_cma_direct_mode_sync_and_async(self) -> None:
        cma = ContextManagerAgent()
        llm = MockLLM(
            default_response="Key Variables:\n- PS1: Inlet Pressure (normal 0-4)"
        )

        state = AgentState(
            user_query="Explain the pressure sensor",
            context=DomainContext(variables={"PS1": "Inlet Pressure"}),
        )

        res = cma.run(state, llm)
        assert isinstance(res, AgentResult)
        assert "Key Variables" in res.output
        assert state.context_summary == res.output
        assert res.trace.execution_time_ms >= 0.0

        # Async run
        async def _run() -> None:
            ares = await cma.arun(state, llm)
            assert "Key Variables" in ares.output

        asyncio.run(_run())

    def test_cma_autonomous_hydration_via_query_params(self) -> None:
        def get_docs(system_id: str) -> dict:
            return {"documents": {"doc1": f"Manual for system {system_id}"}}

        cma = ContextManagerAgent(tools=[get_docs])
        llm = MockLLM(default_response="Structured manual rules.")

        state = AgentState(
            user_query="Status check",
            query_params={"system_id": "hydraulics_01"},
        )
        assert state.context is None

        cma.run(state, llm)
        assert state.context is not None
        assert "Manual for system hydraulics_01" in state.context.documents["doc1"]

    def test_cma_custom_rag_retriever(self) -> None:
        cma = ContextManagerAgent()

        # Set custom callable RAG retriever
        def my_rag(query: str) -> str:
            return f"RAG snippet for query: {query}"

        cma.set_retriever(my_rag)
        assert isinstance(cma.retriever, ContextRetriever)

        llm = MockLLM(default_response="Domain digest using custom RAG.")
        state = AgentState(user_query="How to bypass line?")

        res = cma.run(state, llm)
        assert res.output == "Domain digest using custom RAG."
        assert "RAG snippet for query: How to bypass line?" in llm.last_prompt


# =========================================================================
# 3. Tests for ArtifactsAgent (AA)
# =========================================================================
class TestArtifactsAgent:
    def test_aa_direct_mode_with_prediction_and_xai(self) -> None:
        aa = ArtifactsAgent()
        llm = MockLLM(default_response="Outcome: High Risk (85%). Drivers: PS1 (+0.45)")

        state = AgentState(
            user_query="Why did the alert fire?",
            prediction=PredictionArtifact(
                target_variable="Pump Failure",
                predicted_value="Failure",
                confidence_or_probability=0.85,
            ),
            xai=XAIArtifact(
                feature_importances={"PS1": 0.45, "EPS1": 0.30},
                method="SHAP",
            ),
        )

        res = aa.run(state, llm)
        assert "High Risk" in res.output
        assert state.artifacts_summary == res.output
        assert "Target: Pump Failure" in llm.last_prompt
        assert "PS1: +0.4500" in llm.last_prompt

    def test_aa_zero_ml_mode_skips_llm(self) -> None:
        aa = ArtifactsAgent()
        llm = MockLLM(default_response="Should not be called")

        state = AgentState(user_query="General concept question")
        res = aa.run(state, llm)

        assert "No machine learning prediction" in res.output
        assert llm.call_count == 0  # Zero ML optimization: 0 tokens spent!
        assert "Zero ML mode" in res.trace.summary

    def test_aa_autonomous_query_hydration(self) -> None:
        def fetch_incident_model_data(prediction_id: int) -> dict:
            return {
                "prediction": {
                    "target_variable": "Turbine State",
                    "predicted_value": "Anomaly",
                    "confidence_or_probability": 0.92,
                },
                "xai": {
                    "feature_importances": {"vibration": 0.70},
                },
            }

        aa = ArtifactsAgent(tools=[fetch_incident_model_data])
        llm = MockLLM(default_response="Turbine Anomaly: Vibration driver.")

        state = AgentState(
            user_query="Investigate anomaly",
            query_params={"prediction_id": 99},
        )
        assert state.prediction is None
        assert state.xai is None

        res = aa.run(state, llm)
        assert state.prediction is not None
        assert state.prediction.target_variable == "Turbine State"
        assert state.xai is not None
        assert state.xai.feature_importances["vibration"] == 0.70
        assert res.output == "Turbine Anomaly: Vibration driver."


# =========================================================================
# 4. Tests for ExplainerAgent (EA)
# =========================================================================
class TestExplainerAgent:
    def test_ea_synthesizes_narrative_with_history_and_critique(self) -> None:
        ea = ExplainerAgent()
        llm = MockLLM(default_response="The cooler efficiency dropped due to PS1...")

        history = [
            ChatMessage(role="user", content="What happened earlier?"),
            ChatMessage(role="assistant", content="Normal operation was active."),
        ]

        state = AgentState(
            user_query="Explain the current drop",
            artifacts_summary="Model risk: 82% caused by PS1 (+0.42)",
            context_summary="PS1 is inlet pressure. Limit: 4.0 bar.",
            history=history,
            validation_critique="Fix recommendation clarity.",
        )

        res = ea.run(state, llm)
        assert "cooler efficiency dropped" in res.output
        assert state.draft_explanation == res.output

        # Verify prompt received all contextual inputs
        assert "Model risk: 82%" in llm.last_prompt
        assert "PS1 is inlet pressure" in llm.last_prompt
        assert "Normal operation was active." in llm.last_prompt
        assert "Fix recommendation clarity." in llm.last_prompt
        assert "Explain the current drop" in llm.last_prompt


# =========================================================================
# 5. Tests for ValidatorAgent (VA)
# =========================================================================
class TestValidatorAgent:
    def test_validator_approval_clean_json(self) -> None:
        va = ValidatorAgent(threshold=7.0)
        llm = MockLLM(default_response='{"score": 9.2, "issues": [], "critique": ""}')

        state = AgentState(
            user_query="Explain alert",
            draft_explanation="Cooler pressure exceeded limits.",
        )

        res = va.run(state, llm)
        assert isinstance(res, ValidationResult)
        assert res.score == 9.2
        assert res.passed is True
        assert res.issues == []
        assert res.trace.status == "success"
        assert state.validation_critique is None

    def test_validator_rejection_with_markdown_fences(self) -> None:
        va = ValidatorAgent(threshold=7.5)
        # LLM returning JSON inside markdown code fences
        json_output = """```json
{
  "score": 5.5,
  "issues": ["Hallucinated sensor TS3 which was not in evidence."],
  "critique": "Remove mention of TS3 and adhere strictly to PS1."
}
```"""
        llm = MockLLM(default_response=json_output)

        state = AgentState(
            user_query="Explain alert",
            draft_explanation="Sensor TS3 exploded.",
        )

        res = va.run(state, llm)
        assert isinstance(res, ValidationResult)
        assert res.score == 5.5
        assert res.passed is False
        assert len(res.issues) == 1
        assert "Hallucinated sensor TS3" in res.issues[0]
        assert "Remove mention of TS3" in res.critique
        assert (
            state.validation_critique
            == "Remove mention of TS3 and adhere strictly to PS1."
        )
        assert res.trace.status == "refinement"

    def test_validator_threshold_bounds_and_mutation(self) -> None:
        va = ValidatorAgent(threshold=8.0)
        assert va.threshold == 8.0

        # Mutable threshold by client
        va.threshold = 9.0
        assert va.threshold == 9.0

        with pytest.raises(ValueError, match="between 0.0 and 10.0"):
            ValidatorAgent(threshold=15.0)

        with pytest.raises(ValueError, match="between 0.0 and 10.0"):
            va.threshold = -1.0


# =========================================================================
# 6. Tests for SpecializedAgent (SA)
# =========================================================================
class TestSpecializedAgent:
    def test_specialized_agent_execution(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "turbine_spec.md"
        prompt_file.write_text(
            "Analyze cavitation indicators: {user_query}. Model: {prediction}",
            encoding="utf-8",
        )

        sa = SpecializedAgent(
            name="turbine_expert",
            prompt=prompt_file,
            role="Cavitation Specialist",
        )

        llm = MockLLM(default_response="Cavitation risk is nominal.")
        state = AgentState(
            user_query="Inspect blade vibration",
            prediction=PredictionArtifact(
                target_variable="Blade",
                predicted_value="OK",
            ),
        )

        res = sa.run(state, llm)
        assert res.output == "Cavitation risk is nominal."
        assert (
            state.specialist_insights["turbine_expert"] == "Cavitation risk is nominal."
        )
        assert "Inspect blade vibration" in llm.last_prompt

    def test_specialized_agent_requires_explicit_prompt(self) -> None:
        with pytest.raises(ValueError, match="requires an explicit prompt"):
            SpecializedAgent(name="custom", prompt="")  # type: ignore[arg-type]
