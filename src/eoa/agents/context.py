"""Context Manager Agent (CMA) and Context Retrieval protocols."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eoa.agents.base import AgentResult, AgentState, BaseAgent
from eoa.contracts import AgentStepTrace, DomainContext
from eoa.llm.base import LLM
from eoa.prompts.manager import PromptManager


@runtime_checkable
class ContextRetriever(Protocol):
    """Protocol for pluggable domain context and manual retrieval."""

    def retrieve(self, query: str, context: DomainContext | None = None) -> str:
        """Retrieve relevant context snippets for a user query."""
        ...


class StructuredContextRetriever:
    """Default deterministic retriever for structured schemas, variables, and manuals.

    Requires zero vector databases or heavy external dependencies.
    """

    def retrieve(self, query: str, context: DomainContext | None = None) -> str:
        """Format the DomainContext into a structured technical summary."""
        if context is None or context.is_empty():
            return "No domain context or manual documentation available."

        sections: list[str] = []

        if context.variables:
            vars_str = "\n".join(
                f"- {k}: {v}" for k, v in sorted(context.variables.items())
            )
            sections.append(f"### Variables & Sensor Dictionary\n{vars_str}")

        if context.thresholds:
            thresh_str = "\n".join(
                f"- {k}: {v}" for k, v in sorted(context.thresholds.items())
            )
            sections.append(f"### Operational Thresholds\n{thresh_str}")

        if context.documents:
            docs_str = "\n\n".join(
                f"[{doc_id}]: {text}"
                for doc_id, text in sorted(context.documents.items())
            )
            sections.append(f"### Technical Documentation & Excerpts\n{docs_str}")

        if context.schemas:
            schemas_str = "\n".join(
                f"- {k}: {v}" for k, v in sorted(context.schemas.items())
            )
            sections.append(f"### System Schemas\n{schemas_str}")

        if context.hierarchy:
            hier_str = "\n".join(
                f"- {parent}: {children}"
                for parent, children in sorted(context.hierarchy.items())
            )
            sections.append(f"### Asset Hierarchy\n{hier_str}")

        return "\n\n".join(sections)


class ContextManagerAgent(BaseAgent):
    """Agent responsible for resolving domain dictionaries, schemas, and manuals.

    Can be customized with a pluggable RAG retriever (e.g. LangChain, FAISS,
    LlamaIndex) via ``set_retriever()``.
    """

    def __init__(
        self,
        prompt: str | Path | None = None,
        role: str | None = "Context Manager",
        retriever: ContextRetriever | Callable[[str], str] | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__(
            name="context_manager",
            prompt=prompt,
            role=role,
            tools=tools,
        )
        self.retriever: ContextRetriever = StructuredContextRetriever()
        if retriever:
            self.set_retriever(retriever)

    def set_retriever(self, retriever: ContextRetriever | Callable[[str], str]) -> None:
        """Set a custom domain context retriever (e.g. custom RAG search).

        Args:
            retriever: An instance of ContextRetriever or a callable ``fn(query)``.
        """
        if isinstance(retriever, ContextRetriever):
            self.retriever = retriever
        elif callable(retriever):
            # Adapt simple callable fn(query) into ContextRetriever protocol
            class _CallableRetrieverAdapter:
                def __init__(self, fn: Callable[..., str]) -> None:
                    self.fn = fn
                    self._sig = inspect.signature(fn)

                def retrieve(
                    self, query: str, context: DomainContext | None = None
                ) -> str:
                    if len(self._sig.parameters) >= 2:
                        return self.fn(query, context)
                    return self.fn(query)

            self.retriever = _CallableRetrieverAdapter(retriever)
        else:
            msg = f"Retriever must be ContextRetriever, got {type(retriever).__name__}"
            raise TypeError(msg)

    def _hydrate_context_from_tools_sync(self, state: AgentState) -> None:
        """Execute matching tools to hydrate state.context if missing."""
        if state.context is not None and not state.context.is_empty():
            return

        if not state.query_params or not self.tools:
            return

        results = self._execute_matching_tools_sync(state.query_params)
        for res in results:
            if isinstance(res, DomainContext):
                state.context = state.context.merge(res) if state.context else res
            elif isinstance(res, dict):
                ctx = DomainContext.from_input(res)
                state.context = state.context.merge(ctx) if state.context else ctx

    async def _hydrate_context_from_tools_async(self, state: AgentState) -> None:
        """Asynchronously execute matching tools to hydrate state.context if missing."""
        if state.context is not None and not state.context.is_empty():
            return

        if not state.query_params or not self.tools:
            return

        results = await self._execute_matching_tools_async(state.query_params)
        for res in results:
            if isinstance(res, DomainContext):
                state.context = state.context.merge(res) if state.context else res
            elif isinstance(res, dict):
                ctx = DomainContext.from_input(res)
                state.context = state.context.merge(ctx) if state.context else ctx

    def _build_prompt(self, state: AgentState) -> str:
        """Retrieve context via active retriever and render the CMA prompt."""
        retrieved_text = self.retriever.retrieve(
            query=state.user_query,
            context=state.context,
        )
        return PromptManager.render(
            self.prompt,
            context=retrieved_text,
            user_query=state.user_query,
        )

    def _execute_sync(self, state: AgentState, llm: LLM) -> AgentResult:
        self._hydrate_context_from_tools_sync(state)

        # If zero domain context and zero user query, return gracefully
        if (state.context is None or state.context.is_empty()) and not state.user_query:
            output = "No domain context available."
            state.context_summary = output
            return AgentResult(
                output=output,
                trace=AgentStepTrace(
                    agent_name=self.name,
                    execution_time_ms=0.0,
                    summary="No domain context or query provided.",
                    status="success",
                ),
            )

        rendered_prompt = self._build_prompt(state)
        output = llm.generate(rendered_prompt)
        state.context_summary = output

        return AgentResult(
            output=output,
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Domain context and manual rules structured successfully.",
                status="success",
            ),
        )

    async def _execute_async(self, state: AgentState, llm: LLM) -> AgentResult:
        await self._hydrate_context_from_tools_async(state)

        if (state.context is None or state.context.is_empty()) and not state.user_query:
            output = "No domain context available."
            state.context_summary = output
            return AgentResult(
                output=output,
                trace=AgentStepTrace(
                    agent_name=self.name,
                    execution_time_ms=0.0,
                    summary="No domain context or query provided.",
                    status="success",
                ),
            )

        rendered_prompt = self._build_prompt(state)
        output = await llm.agenerate(rendered_prompt)
        state.context_summary = output

        return AgentResult(
            output=output,
            trace=AgentStepTrace(
                agent_name=self.name,
                execution_time_ms=0.0,
                summary="Domain context and manual rules structured successfully.",
                status="success",
            ),
        )
