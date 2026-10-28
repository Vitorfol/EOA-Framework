"""High-level developer-facing explanation layer and façade for the EOA framework.

Provides the primary interface for developers to interact with the multi-agent
explanation system ("Developer Experience First" & "Developer is King"):
- `ExplanationLayer`: Orchestrator and agent lifecycle façade.
- `default_layer`: 1-line zero-config factory function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eoa.agents.artifacts import ArtifactsAgent
from eoa.agents.base import BaseAgent
from eoa.agents.context import ContextManagerAgent
from eoa.agents.explainer import ExplainerAgent
from eoa.agents.specialized import SpecializedAgent
from eoa.agents.validator import ValidatorAgent
from eoa.contracts import (
    DomainContext,
    ExplanationResponse,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.engine.orchestrator import OrchestratorEngine
from eoa.llm.base import LLM
from eoa.llm.factory import create_llm
from eoa.memory.base import ChatMessage, SessionStore
from eoa.memory.in_memory import InMemorySessionStore
from eoa.memory.sqlite import SQLiteSessionStore

logger = logging.getLogger("eoa.layer")


class ExplanationLayer:
    """Primary developer façade for the Explanation-Oriented Agents (EOA) framework.

    Coordinates agent lifecycles, prompt overrides, conversational memory,
    and explanation execution across Push, Pull, and Multi-Turn scenarios.

    Args:
        llm: Optional LLM instance, custom callable function (Mode 3), or
            pre-configured client (Mode 2). If omitted, resolved via provider/api_key.
        provider: Provider identifier string (e.g., 'gemini').
        api_key: Optional API key string (if not using environment variables).
        model: Model name/identifier (defaults to 'gemini-3.5-flash-lite').
        session_store: Optional SessionStore instance for multi-turn persistence.
            Defaults to an ephemeral `InMemorySessionStore()`. Pass
            `SQLiteSessionStore("path.db")` for file persistence and SQL auditability.
        max_attempts: Maximum refinement iterations in the synthesis/validation loop.
            Defaults to 3.
        auto_create_agents: Whether to automatically instantiate default canonical
            agents (CMA, AA, EA, VA). Defaults to False when instantiating
            `ExplanationLayer` directly, and True when using `eoa.default_layer(...)`.
        **llm_kwargs: Additional parameters forwarded to the LLM constructor or client.

    Example:
        >>> import eoa
        >>> # 1. Direct instantiation
        >>> layer = eoa.ExplanationLayer(provider="gemini", api_key="...")
        >>> layer.create_default_agents()
        >>> response = layer.explain(
        ...     prediction={"target": "Pump", "status": "Degraded", "confidence": 0.88},
        ...     xai={"feature_attributions": {"PS1": 0.52, "EPS1": 0.38}},
        ...     user_query="What caused this degradation?",
        ... )
        >>> print(response.narrative)
    """

    def __init__(
        self,
        llm: LLM | Callable[..., Any] | Any | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        session_store: SessionStore | None = None,
        max_attempts: int = 3,
        auto_create_agents: bool = False,
        **llm_kwargs: Any,
    ) -> None:
        self.llm: LLM = create_llm(
            llm=llm,
            provider=provider,
            api_key=api_key,
            model=model,
            **llm_kwargs,
        )
        self.session_store: SessionStore = (
            session_store if session_store is not None else InMemorySessionStore()
        )
        self.max_attempts: int = max_attempts

        # Internal agent registry keyed by canonical or custom name
        self._agents: dict[str, BaseAgent] = {}
        # Tracks the source of prompt for inspection: "Built-in", Path, or "Inline"
        self._prompt_sources: dict[str, str] = {}
        # Session artifact cache to support fluid multi-turn without re-sending raw data
        self._session_cache: dict[str, dict[str, Any]] = {}

        if auto_create_agents:
            self.create_default_agents()

    def create_default_agents(self, threshold: float = 7.0) -> ExplanationLayer:
        """Instantiate and register the four core canonical agents (CMA, AA, EA, VA).

        Args:
            threshold: Quality Gate acceptance threshold (0.0 to 10.0) for the
                ValidatorAgent. Defaults to 7.0.

        Returns:
            The current `ExplanationLayer` instance for fluent chaining.
        """
        cma = ContextManagerAgent()
        aa = ArtifactsAgent()
        ea = ExplainerAgent()
        va = ValidatorAgent(threshold=threshold)

        self.register_agent(cma, prompt_source="Built-in")
        self.register_agent(aa, prompt_source="Built-in")
        self.register_agent(ea, prompt_source="Built-in")
        self.register_agent(va, prompt_source="Built-in")

        return self

    def register_agent(
        self,
        agent: BaseAgent,
        prompt_source: str | Path | None = None,
    ) -> None:
        """Register or replace an agent in the explanation layer.

        Supports both canonical agents (`context_manager`, `artifacts`, `explainer`,
        `validator`) and custom domain specialists (`SpecializedAgent`).

        Args:
            agent: An instance of `BaseAgent` (or subclasses).
            prompt_source: Optional label or path indicating prompt origin for
                inspection reporting. If omitted, inferred from `agent`.

        Example:
            >>> specialist = SpecializedAgent(
            ...     name="turbine_hydraulic_expert",
            ...     prompt=Path("prompts/turbine.md"),
            ... )
            >>> layer.register_agent(specialist)
        """
        if not isinstance(agent, BaseAgent):
            msg = f"Expected BaseAgent instance, got {type(agent).__name__}"
            raise TypeError(msg)

        self._agents[agent.name] = agent

        if prompt_source is not None:
            self._prompt_sources[agent.name] = str(prompt_source)
        elif agent.name not in self._prompt_sources:
            self._prompt_sources[agent.name] = "Custom instance"

    def remove_agent(self, name: str) -> None:
        """Remove or deactivate an agent by its canonical or custom name.

        Args:
            name: Name of the agent to remove (e.g. 'context_manager', 'artifacts',
                'validator', or custom specialist name).

        Example:
            >>> # Disable Context Manager if domain schemas are not needed:
            >>> layer.remove_agent("context_manager")
        """
        normalized_name = name.strip()
        self._agents.pop(normalized_name, None)
        self._prompt_sources.pop(normalized_name, None)

    def get_agent(self, name: str) -> BaseAgent:
        """Retrieve an active registered agent for inspection or tool/retriever binding.

        Args:
            name: Canonical name of the agent to retrieve.

        Returns:
            The requested `BaseAgent` instance.

        Raises:
            KeyError: If no agent with the given name is currently registered.

        Example:
            >>> cma = layer.get_agent("context_manager")
            >>> cma.set_retriever(my_custom_rag_function)
        """
        normalized_name = name.strip()
        if normalized_name not in self._agents:
            registered = list(self._agents.keys())
            msg = (
                f"Agent '{normalized_name}' not found. "
                f"Registered agents: {registered}"
            )
            raise KeyError(msg)
        return self._agents[normalized_name]

    def override_agent_prompt(
        self, name: str, prompt: str | Path | None
    ) -> None:
        """Surgically override the prompt template of an active registered agent.

        Pass `None` as `prompt` to reset the agent back to its canonical default.

        Args:
            name: Canonical name of the agent (e.g., 'explainer', 'validator').
            prompt: Direct prompt string text, Path to a .md file, or None.

        Raises:
            KeyError: If the agent is not registered.
            ValueError / FileNotFoundError: If the path is invalid or non-markdown.

        Example:
            >>> layer.override_agent_prompt("explainer", Path("prompts/ea.md"))
            >>> # Or reset to default:
            >>> layer.override_agent_prompt("explainer", None)
        """
        agent = self.get_agent(name)
        agent.set_prompt(prompt)
        self._prompt_sources[agent.name] = agent.prompt_source

    def reset_agent_prompt(self, name: str) -> None:
        """Reset an agent's prompt template back to its built-in canonical default.

        Args:
            name: Canonical name of the agent (e.g., 'explainer', 'validator').

        Example:
            >>> layer.reset_agent_prompt("explainer")
        """
        self.override_agent_prompt(name, None)

    def inspect(self, print_output: bool = True) -> str | None:
        """Format and optionally print an inspection table of the layer configuration.

        Displays active agents, their roles, tool counts, and prompt origins.

        Args:
            print_output: If True, prints the formatted table to stdout and
                returns None (ideal for notebooks/scripts). If False, returns the
                formatted table string without printing.

        Returns:
            The formatted table string if `print_output=False`, otherwise None.

        Example:
            >>> layer.inspect()
            >>> # Or capture string without printing:
            >>> table_str = layer.inspect(print_output=False)
        """
        lines = [
            "=" * 78,
            "EOA Framework - Active Explanation Layer Configuration",
            "=" * 78,
            f"{'Agent':<18} {'Role':<24} {'Tools':<8} {'Prompt Source'}",
            "-" * 78,
        ]

        if not self._agents:
            lines.append("No active agents registered in this layer.")
        else:
            for name, agent in self._agents.items():
                role_str = agent.role or "-"
                tools_count = len(agent.tools)
                source_str = (
                    getattr(agent, "prompt_source", None)
                    or self._prompt_sources.get(name, "Unknown")
                )
                lines.append(
                    f"{name:<18} {role_str:<24} {tools_count:<8} {source_str}"
                )

        lines.append("=" * 78)
        output_str = "\n".join(lines)
        if print_output:
            print(output_str)
            return None
        return output_str

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve conversational dialogue turns recorded for a session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of ChatMessage objects in chronological order.

        Example:
            >>> history = layer.get_history("incident_402")
            >>> for msg in history:
            ...     print(f"[{msg.role}]: {msg.content}")
        """
        return self.session_store.get_history(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history and cached artifacts for a session.

        Args:
            session_id: Unique identifier for the conversation session.

        Example:
            >>> layer.clear_session("incident_402")
        """
        self.session_store.clear(session_id)
        self._session_cache.pop(session_id, None)

    def _build_engine(self) -> OrchestratorEngine:
        """Assemble an OrchestratorEngine instance with current registered agents."""
        # Auto-create defaults if no agents are registered at all
        if not self._agents:
            self.create_default_agents()

        cma: ContextManagerAgent | None = None
        aa: ArtifactsAgent | None = None
        ea: ExplainerAgent | None = None
        va: ValidatorAgent | None = None
        specialists: list[SpecializedAgent] = []

        for name, agent in self._agents.items():
            if isinstance(agent, ContextManagerAgent):
                cma = agent
            elif isinstance(agent, ArtifactsAgent):
                aa = agent
            elif isinstance(agent, ExplainerAgent):
                ea = agent
            elif isinstance(agent, ValidatorAgent):
                va = agent
            elif isinstance(agent, SpecializedAgent):
                specialists.append(agent)
            else:
                logger.debug(
                    "Registering custom agent '%s' as specialized participant.", name
                )
                specialists.append(agent)  # type: ignore[arg-type]

        return OrchestratorEngine(
            llm=self.llm,
            cma=cma,
            aa=aa,
            ea=ea,
            va=va,
            specialized_agents=specialists,
            session_store=self.session_store,
            max_attempts=self.max_attempts,
        )

    def _resolve_session_artifacts(
        self,
        session_id: str | None,
        prediction: PredictionArtifact | dict[str, Any] | None,
        xai: XAIArtifact | dict[str, Any] | None,
        context: DomainContext | dict[str, Any] | None,
        query_params: dict[str, Any] | None,
    ) -> tuple[
        PredictionArtifact | dict[str, Any] | None,
        XAIArtifact | dict[str, Any] | None,
        DomainContext | dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """Cache and retrieve artifacts across conversational turns in a session."""
        if not session_id:
            return prediction, xai, context, query_params

        cache = self._session_cache.setdefault(session_id, {})

        if prediction is not None:
            cache["prediction"] = prediction
        if xai is not None:
            cache["xai"] = xai
        if context is not None:
            cache["context"] = context
        if query_params is not None:
            cache["query_params"] = query_params

        resolved_pred = (
            prediction if prediction is not None else cache.get("prediction")
        )
        resolved_xai = xai if xai is not None else cache.get("xai")
        resolved_ctx = context if context is not None else cache.get("context")
        resolved_params = (
            query_params if query_params is not None else cache.get("query_params")
        )

        return resolved_pred, resolved_xai, resolved_ctx, resolved_params

    def _record_validation_if_sqlite(
        self,
        session_id: str,
        response: ExplanationResponse,
    ) -> None:
        """If using SQLiteSessionStore, automatically record audit validation log."""
        if (
            isinstance(self.session_store, SQLiteSessionStore)
            and response.validation_score is not None
        ):
            try:
                self.session_store.log_validation(
                    session_id=session_id,
                    score=response.validation_score,
                    passed=bool(response.validation_passed),
                    critique=response.validation_critique,
                    metadata={
                        "attempts_used": response.attempts_used,
                        "fallback_used": response.fallback_used,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record validation log for session '%s': %s",
                    session_id,
                    exc,
                )

    def explain(
        self,
        user_query: str = "Explain the current system state and prediction.",
        *,
        prediction: PredictionArtifact | dict[str, Any] | None = None,
        xai: XAIArtifact | dict[str, Any] | None = None,
        context: DomainContext | dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        incident_id: str | int | None = None,
        session_id: str | None = None,
    ) -> ExplanationResponse:
        """Generate a validated natural language explanation synchronously.

        Supports three primary integration modes seamlessly:
        1. Direct / Push Mode: Passing `prediction`, `xai`, and/or `context`.
        2. Autonomous / Pull Mode: Passing `incident_id` or `query_params` to
           let registered tools autonomously query model registries and SCADA.
        3. Zero ML / Progressive Ramp: Passing only `user_query` and domain `context`.
        4. Multi-Turn Dialogue: Passing `session_id` to maintain conversation history.

        Args:
            user_query: Operational question or diagnostic prompt.
            prediction: PredictionArtifact object or dictionary (Push mode).
            xai: XAIArtifact object or dictionary (Push mode).
            context: DomainContext object or dictionary (Push mode).
            query_params: Parameters for autonomous tool retrieval (Pull mode).
            incident_id: Convenience identifier injected into `query_params`.
            session_id: Optional session identifier for multi-turn history.

        Returns:
            An `ExplanationResponse` with narrative, validation score, pass flag,
            session ID, and full agent execution trace.

        Example:
            >>> response = layer.explain(
            ...     prediction={"target": "Pump", "status": "Degraded"},
            ...     user_query="Explain the alert.",
            ...     session_id="incident_01",
            ... )
            >>> print(response.narrative)
        """
        engine = self._build_engine()

        res_pred, res_xai, res_ctx, res_params = self._resolve_session_artifacts(
            session_id=session_id,
            prediction=prediction,
            xai=xai,
            context=context,
            query_params=query_params,
        )

        response = engine.run(
            user_query=user_query,
            prediction=res_pred,
            xai=res_xai,
            context=res_ctx,
            query_params=res_params,
            incident_id=incident_id,
            session_id=session_id,
        )

        if session_id and response.session_id:
            self._record_validation_if_sqlite(response.session_id, response)

        return response

    async def aexplain(
        self,
        user_query: str = "Explain the current system state and prediction.",
        *,
        prediction: PredictionArtifact | dict[str, Any] | None = None,
        xai: XAIArtifact | dict[str, Any] | None = None,
        context: DomainContext | dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        incident_id: str | int | None = None,
        session_id: str | None = None,
    ) -> ExplanationResponse:
        """Generate a validated natural language explanation asynchronously.

        Args:
            user_query: Operational question or diagnostic prompt.
            prediction: PredictionArtifact object or dictionary (Push mode).
            xai: XAIArtifact object or dictionary (Push mode).
            context: DomainContext object or dictionary (Push mode).
            query_params: Parameters for autonomous tool retrieval (Pull mode).
            incident_id: Convenience identifier injected into `query_params`.
            session_id: Optional session identifier for multi-turn history.

        Returns:
            An `ExplanationResponse` with narrative, validation score, pass flag,
            session ID, and full agent execution trace.
        """
        engine = self._build_engine()

        res_pred, res_xai, res_ctx, res_params = self._resolve_session_artifacts(
            session_id=session_id,
            prediction=prediction,
            xai=xai,
            context=context,
            query_params=query_params,
        )

        response = await engine.arun(
            user_query=user_query,
            prediction=res_pred,
            xai=res_xai,
            context=res_ctx,
            query_params=res_params,
            incident_id=incident_id,
            session_id=session_id,
        )

        if session_id and response.session_id:
            self._record_validation_if_sqlite(response.session_id, response)

        return response


def default_layer(
    llm: LLM | Callable[..., Any] | Any | None = None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    session_store: SessionStore | None = None,
    threshold: float = 7.0,
    max_attempts: int = 3,
    **llm_kwargs: Any,
) -> ExplanationLayer:
    """One-line factory function to initialize a ready-to-use ExplanationLayer.

    Instantiates the layer and immediately configures the four canonical agents
    (CMA, AA, EA, VA). Fully supports all 3 LLM connection modes:
    - Mode 1: Provider and credentials (`provider="gemini"`, `api_key="..."`).
    - Mode 2: Pre-instantiated Client (`llm=my_client`).
    - Mode 3: Custom callable function (`llm=my_local_llm_function`).

    Args:
        llm: LLM instance, callable function, or third-party client.
        provider: Provider identifier string (e.g. 'gemini').
        api_key: API key string (if not using environment variables).
        model: Model identifier (defaults to 'gemini-3.5-flash-lite').
        session_store: Storage backend for multi-turn sessions (defaults to InMemory).
        threshold: Acceptance threshold for ValidatorAgent (default: 7.0).
        max_attempts: Maximum refinement attempts on failure (default: 3).
        **llm_kwargs: Additional parameters forwarded to LLM creation.

    Returns:
        An initialized `ExplanationLayer` ready to call `.explain()`.

    Examples:
        >>> import eoa
        >>> # Zero-config (reads GEMINI_API_KEY from environment):
        >>> layer = eoa.default_layer()
        >>> response = layer.explain(prediction={...}, user_query="Explain")

        >>> # With custom local callable:
        >>> layer = eoa.default_layer(llm=lambda prompt: "Local model analysis")
    """
    layer = ExplanationLayer(
        llm=llm,
        provider=provider,
        api_key=api_key,
        model=model,
        session_store=session_store,
        max_attempts=max_attempts,
        auto_create_agents=False,
        **llm_kwargs,
    )
    layer.create_default_agents(threshold=threshold)
    return layer
