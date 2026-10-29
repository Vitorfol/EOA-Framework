"""Explainer Agent (EA) for synthesizing natural language explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eoa.agents.base import AgentResult, AgentState, BaseAgent
from eoa.contracts import AgentStepTrace
from eoa.llm.base import LLM
from eoa.memory.base import ChatMessage
from eoa.prompts.manager import PromptManager


class ExplainerAgent(BaseAgent):
    """Synthesizes evidence from AA, CMA, and chat history into an explanation."""

    def __init__(
        self,
        prompt: str | Path | None = None,
        role: str | None = "Explainer",
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__(
            name="explainer",
            prompt=prompt,
            role=role,
            tools=tools,
        )

    def _format_history(self, history: list[ChatMessage]) -> str:
        """Format past conversation turns into a clean dialogue log."""
        if not history:
            return "No previous dialogue history."

        turns = []
        for msg in history:
            turns.append(f"{msg.role.capitalize()}: {msg.content}")
        return "\n".join(turns)

    def _format_domain_context(self, state: AgentState) -> str:
        """Combine CMA context summary with any specialist agent insights."""
        parts: list[str] = []

        if state.context_summary:
            parts.append(f"Domain Context (CMA):\n{state.context_summary}")

        if state.specialist_insights:
            for spec_name, insight in state.specialist_insights.items():
                parts.append(f"Specialist Insight ({spec_name}):\n{insight}")

        if not parts:
            return "No specific domain context or specialist insights provided."

        return "\n\n".join(parts)

    def _build_prompt(
        self,
        state: AgentState,
        tool_outputs: list[Any] | None = None,
    ) -> str:
        artifacts_text = (
            state.artifacts_summary or "No predictive artifacts analysis available."
        )
        domain_text = self._format_domain_context(state)
        history_text = self._format_history(state.history)
        critique_text = (
            state.validation_critique or "None (initial explanation attempt)."
        )
        tools_text = (
            "\n".join(str(item) for item in tool_outputs)
            if tool_outputs
            else "Not provided"
        )

        rendered = PromptManager.render(
            self.prompt,
            artifacts=artifacts_text,
            domain_context=domain_text,
            history=history_text,
            critique=critique_text,
            user_query=state.user_query,
            tool_outputs=tools_text,
        )
        if tool_outputs and "{tool_outputs}" not in self.prompt:
            rendered = f"{rendered.rstrip()}\n\nTool Outputs:\n{tools_text}"
        if "{domain_context}" not in self.prompt and state.specialist_insights:
            rendered = f"{rendered.rstrip()}\n\nSpecialist Insights:\n{domain_text}"
        if "{user_query}" not in self.prompt and state.user_query:
            rendered = f"{rendered.rstrip()}\n\nUser Question:\n{state.user_query}"
        return rendered

    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        output, tool_outputs = self._run_with_tools_sync(
            state,
            llm,
            prompt_builder=lambda tools: self._build_prompt(state, tool_outputs=tools),
        )
        state.draft_explanation = output

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Synthesized narrative explanation for user query.",
                status="success",
            ),
        )

    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        output, tool_outputs = await self._run_with_tools_async(
            state,
            llm,
            prompt_builder=lambda tools: self._build_prompt(state, tool_outputs=tools),
        )
        state.draft_explanation = output

        return AgentResult(
            output=output,
            metadata={"tool_outputs": tool_outputs} if tool_outputs else {},
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Synthesized narrative explanation for user query.",
                status="success",
            ),
        )
