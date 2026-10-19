"""Specialized Agent (SA) for pluggable domain-specific expertise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eoa.agents.base import AgentResult, AgentState, BaseAgent
from eoa.contracts import AgentStepTrace
from eoa.llm.base import LLM
from eoa.prompts.manager import PromptManager


class SpecializedAgent(BaseAgent):
    """Customizable agent dedicated to a specific domain or subsystem.

    Unlike core canonical agents, ``prompt`` is required on SpecializedAgent since
    no built-in default exists for arbitrary user domains.

    Args:
        name: Unique identifier for this specialist (e.g. "turbina_francis").
        prompt: Explicit prompt template string or Path to a local .md file.
        role: Optional descriptive role for telemetry/inspection.
        tools: Optional domain-specific tools or APIs.
    """

    def __init__(
        self,
        name: str,
        prompt: str | Path,
        role: str | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        if not prompt:
            msg = f"SpecializedAgent '{name}' requires an explicit prompt."
            raise ValueError(msg)
        super().__init__(
            name=name,
            prompt=prompt,
            role=role,
            tools=tools,
        )

    def _build_prompt(self, state: AgentState) -> str:
        pred_text = (
            str(state.prediction.model_dump()) if state.prediction else "Not provided"
        )
        xai_text = str(state.xai.model_dump()) if state.xai else "Not provided"
        ctx_text = str(state.context.model_dump()) if state.context else "Not provided"

        return PromptManager.render(
            self.prompt,
            prediction=pred_text,
            xai=xai_text,
            context=ctx_text,
            user_query=state.user_query,
            query_params=str(state.query_params),
        )

    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        # Execute any tools matching query_params if helpful
        if state.query_params and self.tools:
            self._execute_matching_tools_sync(state.query_params)

        rendered_prompt = self._build_prompt(state)
        output = llm.generate(rendered_prompt)
        state.specialist_insights[self.name] = output

        return AgentResult(
            output=output,
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary=f"Specialist '{self.name}' evaluated domain factors.",
                status="success",
            ),
        )

    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        if state.query_params and self.tools:
            await self._execute_matching_tools_async(state.query_params)

        rendered_prompt = self._build_prompt(state)
        output = await llm.agenerate(rendered_prompt)
        state.specialist_insights[self.name] = output

        return AgentResult(
            output=output,
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary=f"Specialist '{self.name}' evaluated domain factors.",
                status="success",
            ),
        )
