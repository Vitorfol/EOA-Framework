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
        temperature: float | None = None,
    ) -> None:
        if not prompt:
            msg = f"SpecializedAgent '{name}' requires an explicit prompt."
            raise ValueError(msg)
        super().__init__(
            name=name,
            prompt=prompt,
            role=role,
            tools=tools,
            temperature=temperature,
        )

    def _build_prompt(
        self,
        state: AgentState,
        tool_outputs: list[Any] | None = None,
    ) -> str:
        pred_text = (
            str(state.prediction.model_dump()) if state.prediction else "Not provided"
        )
        xai_text = str(state.xai.model_dump()) if state.xai else "Not provided"
        ctx_text = str(state.context.model_dump()) if state.context else "Not provided"
        context_summary_text = state.context_summary or "Not provided"
        artifacts_summary_text = state.artifacts_summary or "Not provided"
        tools_text = (
            "\n".join(str(item) for item in tool_outputs)
            if tool_outputs
            else "Not provided"
        )

        rendered = PromptManager.render(
            self.prompt,
            prediction=pred_text,
            xai=xai_text,
            context=ctx_text,
            context_summary=context_summary_text,
            artifacts_summary=artifacts_summary_text,
            tool_outputs=tools_text,
            user_query=state.user_query,
            query_params=str(state.query_params),
        )
        if tool_outputs and "{tool_outputs}" not in self.prompt:
            rendered = f"{rendered}\n\nTool Outputs:\n{tools_text}"
        return rendered

    def _format_trace_summary(self, tool_outputs: list[Any]) -> str:
        if not self.tools:
            return f"Specialist '{self.name}' evaluated domain factors."
        if tool_outputs:
            return (
                f"Specialist '{self.name}' evaluated domain factors "
                f"(tools executed: {tool_outputs})."
            )
        return (
            f"Specialist '{self.name}' evaluated domain factors "
            "(no tools triggered)."
        )

    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        output, tool_outputs = self._run_with_tools_sync(
            state,
            llm,
            prompt_builder=lambda tools: self._build_prompt(state, tool_outputs=tools),
        )
        state.specialist_insights[self.name] = output
        if tool_outputs:
            state.metadata[f"{self.name}_tool_outputs"] = tool_outputs

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary=self._format_trace_summary(tool_outputs),
                status="success",
                details={"tool_outputs": tool_outputs} if tool_outputs else {},
            ),
        )

    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        output, tool_outputs = await self._run_with_tools_async(
            state,
            llm,
            prompt_builder=lambda tools: self._build_prompt(state, tool_outputs=tools),
        )
        state.specialist_insights[self.name] = output
        if tool_outputs:
            state.metadata[f"{self.name}_tool_outputs"] = tool_outputs

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary=self._format_trace_summary(tool_outputs),
                status="success",
                details={"tool_outputs": tool_outputs} if tool_outputs else {},
            ),
        )
