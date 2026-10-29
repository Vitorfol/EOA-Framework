"""Base abstractions, state, and lifecycle protocols for EOA agents."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eoa.contracts import (
    AgentStepTrace,
    DomainContext,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.llm.base import LLM
from eoa.memory.base import ChatMessage
from eoa.prompts.manager import PromptManager
from eoa.tools.base import Tool


class AgentResult(BaseModel):
    """Standardized output produced by an individual agent run.

    Attributes:
        output: Main textual output or synthesis produced by the agent.
        trace: Observability and timing audit trace for this agent's execution step.
        metadata: Optional auxiliary structured data or intermediate artifacts.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    output: str = Field(..., description="Main output or payload from agent execution.")
    trace: AgentStepTrace = Field(
        ...,
        description="Execution trace capturing execution time, status, and summary.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary diagnostic metadata or intermediate data.",
    )


class ValidationResult(AgentResult):
    """Specialized evaluation output produced by the ValidatorAgent.

    Attributes:
        score: Evaluated fidelity score between 0.0 and 10.0.
        passed: Boolean flag indicating if score >= validator threshold.
        issues: List of concrete factual discrepancies or detected hallucinations.
        critique: Directives instructing the explainer how to refine the explanation.
        parse_error: True if the LLM judge response could not be parsed as valid JSON.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Fidelity quality score evaluated by LLM-as-a-Judge.",
    )
    passed: bool = Field(
        ...,
        description="True if score meets or exceeds the validation threshold.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete discrepancies, unverified claims, or hallucinations.",
    )
    critique: str = Field(
        default="",
        description="Actionable guidance to correct the narrative if rejected.",
    )
    parse_error: bool = Field(
        default=False,
        description="True if the validator failed to parse a valid score from the LLM.",
    )


class AgentState(BaseModel):
    """Execution state passed through the agent orchestration pipeline.

    Attributes:
        user_query: The natural language prompt or operational question.
        query_params: Domain-agnostic query parameters and IDs (e.g.,
            ``{"prediction_id": 402, "run_id": "mlflow_99"}``) used by tools.
        prediction: ML model prediction artifact (if available).
        xai: Explainable AI attribution artifact (if available).
        context: Operational context, domain schemas, and manual snippets.
        history: Multi-turn chat message history from SessionStore.
        context_summary: Synthesized domain knowledge produced by CMA.
        artifacts_summary: Synthesized ML/XAI evidence produced by AA.
        specialist_insights: Map of agent name to domain insights produced by SAs.
        draft_explanation: Candidate explanation text produced by EA.
        validation_critique: Auditor critique produced by VA during refinement loops.
        validation_result: Evaluated audit result produced by VA.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user_query: str = Field(..., description="Operational question or user prompt.")
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-agnostic parameters/identifiers for tool queries.",
    )
    prediction: PredictionArtifact | None = Field(
        default=None,
        description="Machine learning prediction artifact.",
    )
    xai: XAIArtifact | None = Field(
        default=None,
        description="Explainable AI attribution artifact.",
    )
    context: DomainContext | None = Field(
        default=None,
        description="Domain schemas, sensor dictionaries, and manual excerpts.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Multi-turn conversation history.",
    )

    # Inter-agent communication blackboard
    context_summary: str | None = Field(
        default=None,
        description="Intermediate domain summary emitted by ContextManagerAgent.",
    )
    artifacts_summary: str | None = Field(
        default=None,
        description="Intermediate ML/XAI summary emitted by ArtifactsAgent.",
    )
    specialist_insights: dict[str, str] = Field(
        default_factory=dict,
        description="Intermediate domain-specific insights keyed by agent name.",
    )
    draft_explanation: str | None = Field(
        default=None,
        description="Draft narrative produced by ExplainerAgent awaiting validation.",
    )
    validation_critique: str | None = Field(
        default=None,
        description="Feedback critique emitted by ValidatorAgent if rejected.",
    )
    validation_result: ValidationResult | None = Field(
        default=None,
        description="Latest structured audit evaluation produced by ValidatorAgent.",
    )


class BaseAgent(ABC):
    """Abstract base class for all EOA agents (CMA, AA, EA, VA, Specialized).

    Handles prompt resolution, tool registration, parameter binding, and
    automatic execution tracing.

    Args:
        name: Canonical identifier of the agent.
        prompt: Explicit prompt template string, Path to a .md file, or None
            to load the built-in default prompt.
        role: Optional descriptive role for telemetry/inspection.
        tools: Optional initial list of Tool objects or Python callables.
    """

    def __init__(
        self,
        name: str,
        prompt: str | Path | None = None,
        role: str | None = None,
        tools: list[Tool | Callable[..., Any]] | None = None,
    ) -> None:
        self.name: str = name
        self.role: str | None = role
        self.prompt: str = PromptManager.resolve(agent=self.name, source=prompt)
        self.prompt_source: str = (
            str(prompt) if isinstance(prompt, Path)
            else ("Inline (str)" if isinstance(prompt, str) else "Built-in")
        )
        self.tools: list[Tool] = []

        if tools:
            for tool_or_fn in tools:
                self.register_tool(tool_or_fn)

    def register_tool(self, tool_or_fn: Tool | Callable[..., Any]) -> Tool:
        """Register a new tool or python callable on this agent.

        Args:
            tool_or_fn: An instance of Tool or any callable Python function.

        Returns:
            The registered Tool instance.
        """
        if isinstance(tool_or_fn, Tool):
            tool = tool_or_fn
        elif callable(tool_or_fn):
            tool = Tool(tool_or_fn)
        else:
            raise TypeError(
                f"Expected Tool or callable, got {type(tool_or_fn).__name__}"
            )

        self.tools.append(tool)
        return tool

    def set_prompt(self, source: str | Path | None) -> None:
        """Update the prompt template for this agent instance.

        Args:
            source: Direct prompt text (str), Path to a local .md file, or None
                to restore the built-in default.
        """
        self.prompt = PromptManager.resolve(agent=self.name, source=source)
        if source is None:
            self.prompt_source = "Built-in"
        elif isinstance(source, Path):
            self.prompt_source = str(source)
        else:
            self.prompt_source = "Inline (str)"

    def get_tool(self, name: str) -> Tool | None:
        """Find a registered tool by its name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _execute_matching_tools_sync(self, query_params: dict[str, Any]) -> list[Any]:
        """Execute all registered tools whose required parameters match query_params."""
        results: list[Any] = []
        for tool in self.tools:
            if tool.can_execute_with(query_params):
                results.append(tool.execute(**query_params))
        return results

    async def _execute_matching_tools_async(
        self, query_params: dict[str, Any]
    ) -> list[Any]:
        """Execute registered tools asynchronously whose params match query_params."""
        results: list[Any] = []
        for tool in self.tools:
            if tool.can_execute_with(query_params):
                res = await tool.aexecute(**query_params)
                results.append(res)
        return results

    @staticmethod
    def _build_tool_selection_prompt(
        user_query: str, candidate_tools: list[Tool]
    ) -> str:
        """Build a prompt instructing the LLM to select tools and extract args."""
        catalog = json.dumps(
            [t.to_descriptor() for t in candidate_tools],
            ensure_ascii=False,
            indent=2,
        )
        template = PromptManager.resolve("tool_dispatch")
        return PromptManager.render(
            template,
            tools_catalog=catalog,
            user_query=user_query,
        )

    @staticmethod
    def _parse_tool_calls(raw_output: str) -> list[dict[str, Any]]:
        """Safely parse tool-call JSON generated by the LLM."""
        if not raw_output:
            return []

        cleaned = raw_output.strip()
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL
        )
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(parsed, dict):
            return []

        calls = parsed.get("tool_calls")
        if not isinstance(calls, list):
            return []

        return [c for c in calls if isinstance(c, dict)]

    @staticmethod
    def _has_empty_tool_calls(raw_output: str) -> bool:
        """Return True if raw_output is a JSON object with an empty tool_calls list."""
        if not raw_output or "tool_calls" not in raw_output:
            return False

        cleaned = raw_output.strip()
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL
        )
        if fence_match:
            cleaned = fence_match.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return False

        return isinstance(parsed, dict) and parsed.get("tool_calls") == []

    def _run_with_tools_sync(
        self,
        state: AgentState,
        llm: LLM,
        prompt_builder: Callable[[list[Any]], str],
        on_tool_results: Callable[[list[Any]], None] | None = None,
    ) -> tuple[str, list[Any]]:
        """Single-pass generic execution with deterministic or autonomous tools.

        If no tool is required by the LLM, the agent answers the main task in the
        first and only LLM call (1 call total). Only when a tool is actively invoked
        does a second LLM call occur with the resulting tool outputs injected.
        """
        if not self.tools:
            return llm.generate(prompt_builder([])), []

        results: list[Any] = []
        remaining_tools: list[Tool] = []

        for tool in self.tools:
            if state.query_params and tool.can_execute_with(state.query_params):
                results.append(tool.execute(**state.query_params))
            else:
                remaining_tools.append(tool)

        if results and on_tool_results is not None:
            on_tool_results(results)

        if not remaining_tools or not state.user_query.strip():
            return llm.generate(prompt_builder(results)), results

        base_prompt = prompt_builder(results)
        dispatch_block = self._build_tool_selection_prompt(
            state.user_query, remaining_tools
        )
        combined_prompt = f"{base_prompt.rstrip()}\n\n{dispatch_block}"
        first_output = llm.generate(combined_prompt)
        tool_calls = self._parse_tool_calls(first_output)

        if not tool_calls:
            if self._has_empty_tool_calls(first_output):
                return llm.generate(base_prompt), results
            return first_output, results

        new_results: list[Any] = []
        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                continue
            target_tool = next(
                (t for t in remaining_tools if t.name == tool_name), None
            )
            if target_tool is None:
                continue
            merged_args = {**state.query_params, **args}
            if target_tool.can_execute_with(merged_args):
                new_results.append(target_tool.execute(**merged_args))

        if new_results:
            results.extend(new_results)
            if on_tool_results is not None:
                on_tool_results(new_results)
            return llm.generate(prompt_builder(results)), results

        return llm.generate(base_prompt), results

    async def _run_with_tools_async(
        self,
        state: AgentState,
        llm: LLM,
        prompt_builder: Callable[[list[Any]], str],
        on_tool_results: Callable[[list[Any]], None] | None = None,
    ) -> tuple[str, list[Any]]:
        """Asynchronously execute single-pass agent generation with tools."""
        if not self.tools:
            return await llm.agenerate(prompt_builder([])), []

        results: list[Any] = []
        remaining_tools: list[Tool] = []

        for tool in self.tools:
            if state.query_params and tool.can_execute_with(state.query_params):
                res = await tool.aexecute(**state.query_params)
                results.append(res)
            else:
                remaining_tools.append(tool)

        if results and on_tool_results is not None:
            on_tool_results(results)

        if not remaining_tools or not state.user_query.strip():
            return await llm.agenerate(prompt_builder(results)), results

        base_prompt = prompt_builder(results)
        dispatch_block = self._build_tool_selection_prompt(
            state.user_query, remaining_tools
        )
        combined_prompt = f"{base_prompt.rstrip()}\n\n{dispatch_block}"
        first_output = await llm.agenerate(combined_prompt)
        tool_calls = self._parse_tool_calls(first_output)

        if not tool_calls:
            if self._has_empty_tool_calls(first_output):
                return await llm.agenerate(base_prompt), results
            return first_output, results

        new_results: list[Any] = []
        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                continue
            target_tool = next(
                (t for t in remaining_tools if t.name == tool_name), None
            )
            if target_tool is None:
                continue
            merged_args = {**state.query_params, **args}
            if target_tool.can_execute_with(merged_args):
                res = await target_tool.aexecute(**merged_args)
                new_results.append(res)

        if new_results:
            results.extend(new_results)
            if on_tool_results is not None:
                on_tool_results(new_results)
            return await llm.agenerate(prompt_builder(results)), results

        return await llm.agenerate(base_prompt), results

    def _resolve_and_execute_tools_sync(
        self, state: AgentState, llm: LLM
    ) -> list[Any]:
        """Execute tools via deterministic query_params fast-path or LLM reasoning."""
        if not self.tools:
            return []

        results: list[Any] = []
        remaining_tools: list[Tool] = []

        for tool in self.tools:
            if state.query_params and tool.can_execute_with(state.query_params):
                results.append(tool.execute(**state.query_params))
            else:
                remaining_tools.append(tool)

        if not remaining_tools or not state.user_query.strip():
            return results

        selection_prompt = self._build_tool_selection_prompt(
            state.user_query, remaining_tools
        )
        raw_decision = llm.generate(selection_prompt)
        tool_calls = self._parse_tool_calls(raw_decision)

        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                continue
            target_tool = next(
                (t for t in remaining_tools if t.name == tool_name), None
            )
            if target_tool is None:
                continue
            merged_args = {**state.query_params, **args}
            if target_tool.can_execute_with(merged_args):
                results.append(target_tool.execute(**merged_args))

        return results

    async def _resolve_and_execute_tools_async(
        self, state: AgentState, llm: LLM
    ) -> list[Any]:
        """Asynchronously execute tools via query_params fast-path or LLM reasoning."""
        if not self.tools:
            return []

        results: list[Any] = []
        remaining_tools: list[Tool] = []

        for tool in self.tools:
            if state.query_params and tool.can_execute_with(state.query_params):
                res = await tool.aexecute(**state.query_params)
                results.append(res)
            else:
                remaining_tools.append(tool)

        if not remaining_tools or not state.user_query.strip():
            return results

        selection_prompt = self._build_tool_selection_prompt(
            state.user_query, remaining_tools
        )
        raw_decision = await llm.agenerate(selection_prompt)
        tool_calls = self._parse_tool_calls(raw_decision)

        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                continue
            target_tool = next(
                (t for t in remaining_tools if t.name == tool_name), None
            )
            if target_tool is None:
                continue
            merged_args = {**state.query_params, **args}
            if target_tool.can_execute_with(merged_args):
                res = await target_tool.aexecute(**merged_args)
                results.append(res)

        return results

    @abstractmethod
    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        """Internal synchronous execution logic implemented by subclasses."""
        ...

    @abstractmethod
    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        """Internal asynchronous execution logic implemented by subclasses."""
        ...

    def run(self, state: AgentState, llm: LLM) -> AgentResult:
        """Execute the agent synchronously and record latency in the trace.

        Args:
            state: Shared pipeline state.
            llm: Language model abstraction for text generation.

        Returns:
            An AgentResult (or ValidationResult) instance.
        """
        start_time = time.perf_counter()
        result = self._execute_sync(state=state, llm=llm)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        result.trace.execution_time_ms = elapsed_ms
        return result

    async def arun(self, state: AgentState, llm: LLM) -> AgentResult:
        """Execute the agent asynchronously and record latency in the trace.

        Args:
            state: Shared pipeline state.
            llm: Language model abstraction for text generation.

        Returns:
            An AgentResult (or ValidationResult) instance.
        """
        start_time = time.perf_counter()
        result = await self._execute_async(state=state, llm=llm)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        result.trace.execution_time_ms = elapsed_ms
        return result
