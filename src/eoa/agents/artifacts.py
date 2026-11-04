"""Artifacts Agent (AA) for interpreting ML predictions and XAI attributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eoa.agents.base import AgentResult, AgentState, BaseAgent
from eoa.contracts import (
    AgentStepTrace,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.llm.base import LLM
from eoa.prompts.manager import PromptManager


class ArtifactsAgent(BaseAgent):
    """Interprets model predictions and feature importance attributions (XAI).

    Supports:
    - Direct Mode: Consumes state.prediction and state.xai.
    - Autonomous Query Mode: Fetches prediction/XAI via tools matching query_params.
    - Zero ML Mode: Returns immediately without LLM if no ML artifacts exist.
    """

    def __init__(
        self,
        prompt: str | Path | None = None,
        role: str | None = "Artifacts Analyst",
        tools: list[Any] | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(
            name="artifacts",
            prompt=prompt,
            role=role,
            tools=tools,
            temperature=temperature,
        )

    def _hydrate_artifacts_from_tools_sync(
        self, state: AgentState, llm: LLM
    ) -> list[Any]:
        """Hydrate prediction and xai from tools if missing."""
        if state.prediction is not None or not self.tools:
            return []

        results = self._resolve_and_execute_tools_sync(state, llm)
        for res in results:
            self._apply_tool_result(state, res)
        return results

    async def _hydrate_artifacts_from_tools_async(
        self, state: AgentState, llm: LLM
    ) -> list[Any]:
        """Asynchronously hydrate prediction and xai from tools if missing."""
        if state.prediction is not None or not self.tools:
            return []

        results = await self._resolve_and_execute_tools_async(state, llm)
        for res in results:
            self._apply_tool_result(state, res)
        return results

    @staticmethod
    def _apply_tool_result(state: AgentState, res: Any) -> None:
        """Inspect tool result and populate state.prediction / state.xai."""
        if isinstance(res, PredictionArtifact):
            state.prediction = res
        elif isinstance(res, XAIArtifact):
            state.xai = res
        elif isinstance(res, dict):
            # Combined dict or single artifact
            if "prediction" in res:
                pred_val = res["prediction"]
                state.prediction = (
                    pred_val
                    if isinstance(pred_val, PredictionArtifact)
                    else PredictionArtifact.model_validate(pred_val)
                )
            if "xai" in res:
                xai_val = res["xai"]
                state.xai = (
                    xai_val
                    if isinstance(xai_val, XAIArtifact)
                    else XAIArtifact.model_validate(xai_val)
                )
            # Standalone prediction dictionary
            if state.prediction is None and (
                "target_variable" in res or "predicted_value" in res
            ):
                state.prediction = PredictionArtifact.model_validate(res)
            # Standalone XAI dictionary
            if state.xai is None and "feature_attributions" in res:
                state.xai = XAIArtifact.model_validate(res)

    def _format_prediction_text(self, prediction: PredictionArtifact | None) -> str:
        if prediction is None:
            return "No machine learning prediction provided."

        details = [
            f"Target: {prediction.target_variable}",
            f"Predicted Value: {prediction.predicted_value}",
        ]
        if prediction.confidence_or_probability is not None:
            details.append(
                f"Confidence/Risk: {prediction.confidence_or_probability:.2%}"
            )
        if prediction.status_label:
            details.append(f"Status: {prediction.status_label}")
        if prediction.model_name:
            details.append(f"Model: {prediction.model_name}")
        if prediction.metadata:
            details.append(f"Metadata: {prediction.metadata}")

        return "\n".join(details)

    def _format_xai_text(self, xai: XAIArtifact | None) -> str:
        if xai is None or not xai.feature_attributions:
            return (
                "Feature attributions (XAI) not provided; "
                "relying strictly on prediction outcome."
            )

        lines = ["Ranked Feature Attributions:"]
        for feat, score in xai.top_features(10):
            lines.append(f"- {feat}: {score:+.4f}")

        if xai.method:
            lines.append(f"Attribution Method: {xai.method}")

        return "\n".join(lines)

    def _build_prompt(self, state: AgentState) -> str:
        pred_text = self._format_prediction_text(state.prediction)
        xai_text = self._format_xai_text(state.xai)

        return PromptManager.render(
            self.prompt,
            prediction=pred_text,
            xai=xai_text,
        )

    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        tool_outputs = self._hydrate_artifacts_from_tools_sync(state, llm)
        if tool_outputs:
            state.metadata[f"{self.name}_tool_outputs"] = tool_outputs

        # Zero ML / Progressive Ramp optimization: save tokens if no prediction exists
        if state.prediction is None:
            output = "No machine learning prediction or XAI artifacts provided."
            state.artifacts_summary = output
            return AgentResult(
                output=output,
                metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
                trace=AgentStepTrace(
                    agent_name=self.name,
                    execution_time_ms=0.0,
                    summary="Zero ML mode: Skipped LLM evaluation without prediction.",
                    status="success",
                    details={"tool_outputs": tool_outputs} if tool_outputs else {},
                ),
            )

        rendered_prompt = self._build_prompt(state)
        output = self._call_llm_sync(llm, rendered_prompt)
        state.artifacts_summary = output

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Prediction and XAI feature attributions analyzed.",
                status="success",
                details={"tool_outputs": tool_outputs} if tool_outputs else {},
            ),
        )

    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        tool_outputs = await self._hydrate_artifacts_from_tools_async(state, llm)
        if tool_outputs:
            state.metadata[f"{self.name}_tool_outputs"] = tool_outputs

        # Zero ML / Progressive Ramp optimization
        if state.prediction is None:
            output = "No machine learning prediction or XAI artifacts provided."
            state.artifacts_summary = output
            return AgentResult(
                output=output,
                metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
                trace=AgentStepTrace(
                    agent_name=self.name,
                    execution_time_ms=0.0,
                    summary="Zero ML mode: Skipped LLM evaluation without prediction.",
                    status="success",
                    details={"tool_outputs": tool_outputs} if tool_outputs else {},
                ),
            )

        rendered_prompt = self._build_prompt(state)
        output = await self._call_llm_async(llm, rendered_prompt)
        state.artifacts_summary = output

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Prediction and XAI feature attributions analyzed.",
                status="success",
                details={"tool_outputs": tool_outputs} if tool_outputs else {},
            ),
        )
