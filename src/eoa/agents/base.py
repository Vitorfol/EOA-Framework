"""Base abstractions, state, and lifecycle protocols for EOA agents."""

from __future__ import annotations

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
