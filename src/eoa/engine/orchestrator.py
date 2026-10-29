"""Orchestrator Engine, lightweight state machine, and fail-safe execution for EOA.

This module implements the core execution graph of the EOA architecture:
1. Session History Retrieval (from SessionStore if session_id is explicitly provided).
2. Phase 1 (Evidence Gathering): Concurrent execution of CMA, AA, and SpecializedAgents.
3. Phase 2 (Synthesis & Quality Gate): Iterative loop between ExplainerAgent (EA) and
   ValidatorAgent (VA) up to `max_attempts` refinements.
4. Unplugged Fail-Safe Mode: Provider-agnostic exception handling and deterministic
   fallback synthesis ensuring host systems never crash due to LLM/network failures.
5. Session History Persistence (saving user query and response turns to SessionStore).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eoa.agents.artifacts import ArtifactsAgent
from eoa.agents.base import AgentResult, AgentState, BaseAgent, ValidationResult
from eoa.agents.context import ContextManagerAgent
from eoa.agents.explainer import ExplainerAgent
from eoa.agents.specialized import SpecializedAgent
from eoa.agents.validator import ValidatorAgent
from eoa.contracts import (
    AgentStepTrace,
    DomainContext,
    ExplanationResponse,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.llm.base import LLM
from eoa.memory.base import ROLE_ASSISTANT, ROLE_USER, ChatMessage, SessionStore

logger = logging.getLogger("eoa.orchestrator")


class OrchestratorEngine:
    """Coordinates the multi-agent explanation workflow and quality gate loop.

    Executes the ContextManagerAgent (CMA), ArtifactsAgent (AA), and any registered
    SpecializedAgents concurrently, followed by the ExplainerAgent (EA) and
    ValidatorAgent (VA) refinement loop. Incorporates an "Unplugged" fail-safe
    mechanism that gracefully degrades to a deterministic technical summary if
    the underlying LLM provider or validation pipeline fails.

    Args:
        llm: Unified LLM provider instance.
        cma: Optional ContextManagerAgent instance.
        aa: Optional ArtifactsAgent instance.
        ea: Optional ExplainerAgent instance (created by default if omitted).
        va: Optional ValidatorAgent instance.
        specialized_agents: Optional list of domain-specific SpecializedAgents.
        session_store: Optional SessionStore for multi-turn conversation persistence.
        max_attempts: Maximum synthesis/refinement attempts (>= 1). Defaults to 3.

    Example:
        >>> from eoa import OrchestratorEngine, ContextManagerAgent, ArtifactsAgent
        >>> from eoa import ExplainerAgent, ValidatorAgent
        >>> engine = OrchestratorEngine(
        ...     llm=my_llm,
        ...     cma=ContextManagerAgent(),
        ...     aa=ArtifactsAgent(),
        ...     ea=ExplainerAgent(),
        ...     va=ValidatorAgent(threshold=7.0),
        ... )
        >>> response = engine.run(user_query="Why did alert 402 fire?")
    """

    def __init__(
        self,
        llm: LLM,
        cma: ContextManagerAgent | None = None,
        aa: ArtifactsAgent | None = None,
        ea: ExplainerAgent | None = None,
        va: ValidatorAgent | None = None,
        specialized_agents: list[SpecializedAgent] | None = None,
        session_store: SessionStore | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        self.llm: LLM = llm
        self.cma: ContextManagerAgent | None = cma
        self.aa: ArtifactsAgent | None = aa
        self.ea: ExplainerAgent | None = ea
        self.va: ValidatorAgent | None = va
        self.specialized_agents: list[SpecializedAgent] = list(
            specialized_agents or []
        )
        self.session_store: SessionStore | None = session_store
        self.max_attempts: int = max_attempts
        self.last_state: AgentState | None = None

    @staticmethod
    def _normalize_inputs(
        user_query: str,
        prediction: PredictionArtifact | dict[str, Any] | None,
        xai: XAIArtifact | dict[str, Any] | None,
        context: DomainContext | dict[str, Any] | None,
        query_params: dict[str, Any] | None,
        incident_id: str | int | None,
        history: list[ChatMessage],
    ) -> AgentState:
        """Coerce raw inputs into a strictly typed AgentState instance."""
        pred_obj = (
            PredictionArtifact.from_input(prediction)
            if prediction is not None
            else None
        )
        xai_obj = XAIArtifact.from_input(xai) if xai is not None else None
        ctx_obj = DomainContext.from_input(context) if context is not None else None

        merged_params: dict[str, Any] = dict(query_params or {})
        if incident_id is not None and "incident_id" not in merged_params:
            merged_params["incident_id"] = incident_id

        return AgentState(
            user_query=user_query,
            query_params=merged_params,
            prediction=pred_obj,
            xai=xai_obj,
            context=ctx_obj,
            history=history,
        )

    @staticmethod
    def _build_deterministic_fallback(
        state: AgentState,
        reason: str,
    ) -> str:
        """Synthesize a deterministic, non-LLM fallback summary from raw state data.

        Guarantees that mission-critical host systems receive actionable analytical
        evidence even during complete LLM outages, network timeouts, or repeated
        quality gate failures ("Unplugged" fail-safe mode).
        """
        lines: list[str] = [
            "[FALLBACK DETERMINÍSTICO ATIVADO / DEGRADAÇÃO GRACIOSA]",
            f"Motivo Técnico: {reason}",
            "",
        ]

        if state.user_query:
            lines.append(f"Consulta Operacional: {state.user_query}")

        # 1. Prediction evidence
        if state.prediction is not None:
            pred = state.prediction
            lines.append("")
            lines.append("1. Resumo da Predição Analítica (Nível 1):")
            lines.append(f"   - Variável Alvo: {pred.target_variable}")
            lines.append(f"   - Valor Predito: {pred.predicted_value}")
            if pred.confidence_or_probability is not None:
                lines.append(
                    f"   - Confiança / Probabilidade: "
                    f"{pred.confidence_or_probability:.2%}"
                )
            if pred.status_label:
                lines.append(f"   - Status: {pred.status_label}")
            if pred.model_name:
                lines.append(f"   - Modelo: {pred.model_name}")
        else:
            lines.append("")
            lines.append(
                "1. Resumo da Predição Analítica: Nenhuma predição disponível."
            )

        # 2. XAI feature attributions
        if state.xai is not None and state.xai.feature_attributions:
            lines.append("")
            lines.append(
                f"2. Principais Fatores de Explicabilidade "
                f"(Nível 2 - {state.xai.method}):"
            )
            for feat, weight in state.xai.top_features(5):
                lines.append(f"   - {feat}: {weight:+.4f}")
        else:
            lines.append("")
            lines.append("2. Atribuições XAI: Não fornecidas.")

        # 3. Domain context & operational rules
        if state.context is not None and not state.context.is_empty():
            ctx = state.context
            lines.append("")
            lines.append("3. Contexto Operacional e Regras de Domínio:")
            if ctx.variables:
                vars_str = ", ".join(
                    f"{k} ({v})" for k, v in sorted(ctx.variables.items())
                )
                lines.append(f"   - Variáveis: {vars_str}")
            if ctx.thresholds:
                thresh_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(ctx.thresholds.items())
                )
                lines.append(f"   - Limiares: {thresh_str}")
            if ctx.documents:
                for doc_key, doc_text in sorted(ctx.documents.items()):
                    lines.append(f"   - Documento [{doc_key}]: {doc_text}")
        elif state.context_summary:
            lines.append("")
            lines.append(f"3. Contexto Recuperado: {state.context_summary}")

        # 4. Specialist insights if available
        if state.specialist_insights:
            lines.append("")
            lines.append("4. Pareceres de Agentes Especialistas:")
            for name, insight in state.specialist_insights.items():
                lines.append(f"   - [{name}]: {insight}")

        return "\n".join(lines)

    def _collect_phase1_agents(self) -> list[BaseAgent]:
        """Return active Phase 1 evidence-gathering agents (CMA, AA, SAs)."""
        agents: list[BaseAgent] = []
        if self.cma is not None:
            agents.append(self.cma)
        if self.aa is not None:
            agents.append(self.aa)
        agents.extend(self.specialized_agents)
        return agents

    def run(
        self,
        user_query: str = "Explain the current system state and prediction.",
        prediction: PredictionArtifact | dict[str, Any] | None = None,
        xai: XAIArtifact | dict[str, Any] | None = None,
        context: DomainContext | dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        incident_id: str | int | None = None,
        session_id: str | None = None,
    ) -> ExplanationResponse:
        """Execute the full explanation and validation workflow synchronously.

        Args:
            user_query: The question or prompt from the operator/user.
            prediction: Direct ML prediction artifact or dict (Push mode).
            xai: Direct XAI feature attribution artifact or dict (Push mode).
            context: Direct domain context or dict (Push mode).
            query_params: Parameters/identifiers for autonomous tool queries
                (Pull mode).
            incident_id: Convenience identifier injected into ``query_params``.
            session_id: Optional conversation session identifier. If provided and a
                ``session_store`` is configured, loads prior history and persists
                the new turn. If omitted, runs statelessly with a generated UUID.

        Returns:
            A validated ``ExplanationResponse`` containing the narrative, validation
            metrics, session ID, and complete execution trace.
        """
        pipeline_start = time.perf_counter()
        explicit_session = session_id is not None and len(session_id.strip()) > 0
        resolved_session_id = (
            session_id.strip() if explicit_session and session_id else uuid.uuid4().hex
        )

        logger.info(
            "Starting synchronous orchestration pipeline "
            "(session_id=%s, persistent=%s)",
            resolved_session_id,
            bool(explicit_session and self.session_store is not None),
        )

        # 1. Load conversation history if explicit session_id + session_store
        history: list[ChatMessage] = []
        if explicit_session and self.session_store is not None:
            try:
                history = self.session_store.get_history(resolved_session_id)
                logger.debug(
                    "Loaded %d prior messages for session '%s'.",
                    len(history),
                    resolved_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to retrieve history for session '%s': %s",
                    resolved_session_id,
                    exc,
                )

        state = self._normalize_inputs(
            user_query=user_query,
            prediction=prediction,
            xai=xai,
            context=context,
            query_params=query_params,
            incident_id=incident_id,
            history=history,
        )
        self.last_state = state

        traces: list[AgentStepTrace] = []
        attempts_used = 1
        fallback_used = False
        validation_score: float | None = None
        validation_passed: bool | None = None
        validation_critique: str | None = None
        best_narrative: str = ""
        best_score: float = -1.0
        final_narrative: str = ""

        try:
            # 2. Phase 1: Concurrent execution of CMA, AA, and SpecializedAgents
            phase1_agents = self._collect_phase1_agents()
            if phase1_agents:
                logger.info(
                    "Executing Phase 1 evidence agents concurrently: %s",
                    [a.name for a in phase1_agents],
                )
                if len(phase1_agents) == 1:
                    res = phase1_agents[0].run(state, self.llm)
                    traces.append(res.trace)
                else:
                    with ThreadPoolExecutor(max_workers=len(phase1_agents)) as pool:
                        futures = [
                            pool.submit(agent.run, state, self.llm)
                            for agent in phase1_agents
                        ]
                        for fut in futures:
                            res = fut.result()
                            traces.append(res.trace)

            # 3. Phase 2: Synthesis (EA) & Quality Gate Refinement Loop (VA)
            if self.ea is None:
                logger.info(
                    "No ExplainerAgent configured; synthesizing deterministic summary."
                )
                final_narrative = self._build_deterministic_fallback(
                    state, reason="ExplainerAgent (EA) não configurado na camada."
                )
                fallback_used = True
            else:
                effective_max_attempts = (
                    self.max_attempts if self.va is not None else 1
                )

                for attempt in range(1, effective_max_attempts + 1):
                    attempts_used = attempt
                    logger.info(
                        "Running ExplainerAgent synthesis (attempt %d/%d).",
                        attempt,
                        effective_max_attempts,
                    )
                    ea_result = self.ea.run(state, self.llm)
                    traces.append(ea_result.trace)
                    current_narrative = ea_result.output
                    final_narrative = current_narrative

                    if self.va is None:
                        break

                    logger.info(
                        "Running ValidatorAgent quality gate (attempt %d/%d).",
                        attempt,
                        effective_max_attempts,
                    )
                    va_result = self.va.run(state, self.llm)
                    traces.append(va_result.trace)

                    if isinstance(va_result, ValidationResult):
                        validation_score = va_result.score
                        validation_passed = va_result.passed
                        validation_critique = va_result.critique

                        # Track best candidate narrative observed across attempts
                        if va_result.score > best_score:
                            best_score = va_result.score
                            best_narrative = current_narrative

                        if va_result.parse_error:
                            logger.error(
                                "ValidatorAgent encountered a JSON parse error on "
                                "attempt %d. Activating deterministic fail-safe.",
                                attempt,
                            )
                            fallback_used = True
                            validation_passed = False
                            final_narrative = self._build_deterministic_fallback(
                                state,
                                reason=(
                                    "Falha de formato (parse_error) na auditoria "
                                    "do ValidatorAgent."
                                ),
                            )
                            traces.append(
                                AgentStepTrace(
                                    agent_name="orchestrator",
                                    execution_time_ms=0.0,
                                    summary=(
                                        "Fail-safe fallback triggered due to "
                                        "validator JSON parse error."
                                    ),
                                    status="fallback",
                                    details={"attempt": attempt},
                                )
                            )
                            break

                        if va_result.passed:
                            logger.info(
                                "Explanation approved by ValidatorAgent on attempt %d "
                                "(score=%.2f >= threshold=%.2f).",
                                attempt,
                                va_result.score,
                                self.va.threshold,
                            )
                            break

                        logger.warning(
                            "ValidatorAgent rejected draft on attempt %d/%d "
                            "(score=%.2f < threshold=%.2f). Critique: %s",
                            attempt,
                            effective_max_attempts,
                            va_result.score,
                            self.va.threshold,
                            va_result.critique,
                        )

                # Check if refinement loop exhausted without approval
                if (
                    self.va is not None
                    and validation_passed is False
                    and not fallback_used
                ):
                    logger.warning(
                        "Refinement loop exhausted after %d attempts (best_score=%.2f, "
                        "last_score=%.2f). Delivering best candidate narrative with "
                        "quality warning.",
                        attempts_used,
                        best_score if best_score >= 0.0 else 0.0,
                        validation_score or 0.0,
                    )
                    fallback_used = True
                    # Keep best candidate narrative and align score with delivered text
                    if best_narrative:
                        final_narrative = best_narrative
                    if best_score >= 0.0:
                        validation_score = best_score

                    summary_msg = (
                        f"Quality gate threshold not met after {attempts_used} "
                        f"attempts (delivered best_score={validation_score}). "
                        f"Delivered best draft with low-confidence indicator."
                    )
                    traces.append(
                        AgentStepTrace(
                            agent_name="orchestrator",
                            execution_time_ms=0.0,
                            summary=summary_msg,
                            status="fallback",
                            details={
                                "attempts_used": attempts_used,
                                "best_score": best_score,
                                "last_score": va_result.score
                                if isinstance(va_result, ValidationResult)
                                else None,
                            },
                        )
                    )

        except Exception as exc:
            # 4. Unplugged Fail-Safe Mode on any unexpected LLM / tool / runtime failure
            elapsed_err_ms = (time.perf_counter() - pipeline_start) * 1000.0
            logger.error(
                "Orchestrator caught exception (%s: %s). Activating Unplugged "
                "fail-safe mode.",
                type(exc).__name__,
                exc,
            )
            fallback_used = True
            if self.va is not None:
                validation_passed = False
            final_narrative = self._build_deterministic_fallback(
                state,
                reason=f"Exceção interceptada ({type(exc).__name__}: {exc})",
            )
            traces.append(
                AgentStepTrace(
                    agent_name="orchestrator",
                    execution_time_ms=elapsed_err_ms,
                    summary=(
                        f"Unplugged fail-safe activated due to "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    status="fallback",
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )

        # 5. Persist conversation turn if explicit session_id + session_store
        if explicit_session and self.session_store is not None:
            try:
                user_msg = ChatMessage(
                    role=ROLE_USER,
                    content=user_query,
                    metadata={"session_id": resolved_session_id},
                )
                assistant_msg = ChatMessage(
                    role=ROLE_ASSISTANT,
                    content=final_narrative,
                    metadata={
                        "session_id": resolved_session_id,
                        "validation_score": validation_score,
                        "validation_passed": validation_passed,
                        "attempts_used": attempts_used,
                        "fallback_used": fallback_used,
                    },
                )
                self.session_store.append_message(resolved_session_id, user_msg)
                self.session_store.append_message(resolved_session_id, assistant_msg)
                logger.debug(
                    "Persisted user and assistant messages to session '%s'.",
                    resolved_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist messages for session '%s': %s",
                    resolved_session_id,
                    exc,
                )

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        logger.info(
            "Orchestration completed in %.2fms (attempts=%d, fallback=%s, score=%s).",
            total_ms,
            attempts_used,
            fallback_used,
            validation_score,
        )

        return ExplanationResponse(
            narrative=final_narrative,
            validation_score=validation_score,
            validation_passed=validation_passed,
            validation_critique=validation_critique,
            attempts_used=attempts_used,
            fallback_used=fallback_used,
            session_id=resolved_session_id,
            execution_trace=traces,
        )

    async def arun(
        self,
        user_query: str = "Explain the current system state and prediction.",
        prediction: PredictionArtifact | dict[str, Any] | None = None,
        xai: XAIArtifact | dict[str, Any] | None = None,
        context: DomainContext | dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        incident_id: str | int | None = None,
        session_id: str | None = None,
    ) -> ExplanationResponse:
        """Execute the full explanation and validation workflow asynchronously.

        Args:
            user_query: The question or prompt from the operator/user.
            prediction: Direct ML prediction artifact or dict (Push mode).
            xai: Direct XAI feature attribution artifact or dict (Push mode).
            context: Direct domain context or dict (Push mode).
            query_params: Parameters/identifiers for autonomous tool queries
                (Pull mode).
            incident_id: Convenience identifier injected into ``query_params``.
            session_id: Optional conversation session identifier. If provided and a
                ``session_store`` is configured, loads prior history and persists
                the new turn. If omitted, runs statelessly with a generated UUID.

        Returns:
            A validated ``ExplanationResponse`` containing the narrative, validation
            metrics, session ID, and complete execution trace.
        """
        pipeline_start = time.perf_counter()
        explicit_session = session_id is not None and len(session_id.strip()) > 0
        resolved_session_id = (
            session_id.strip() if explicit_session and session_id else uuid.uuid4().hex
        )

        logger.info(
            "Starting asynchronous orchestration pipeline "
            "(session_id=%s, persistent=%s)",
            resolved_session_id,
            bool(explicit_session and self.session_store is not None),
        )

        # 1. Load conversation history if explicit session_id + session_store
        history: list[ChatMessage] = []
        if explicit_session and self.session_store is not None:
            try:
                history = await self.session_store.aget_history(resolved_session_id)
                logger.debug(
                    "Loaded %d prior messages for session '%s'.",
                    len(history),
                    resolved_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to retrieve history for session '%s': %s",
                    resolved_session_id,
                    exc,
                )

        state = self._normalize_inputs(
            user_query=user_query,
            prediction=prediction,
            xai=xai,
            context=context,
            query_params=query_params,
            incident_id=incident_id,
            history=history,
        )
        self.last_state = state

        traces: list[AgentStepTrace] = []
        attempts_used = 1
        fallback_used = False
        validation_score: float | None = None
        validation_passed: bool | None = None
        validation_critique: str | None = None
        best_narrative: str = ""
        best_score: float = -1.0
        final_narrative: str = ""

        try:
            # 2. Phase 1: Concurrent execution of CMA, AA, and SpecializedAgents
            phase1_agents = self._collect_phase1_agents()
            if phase1_agents:
                logger.info(
                    "Executing Phase 1 evidence agents concurrently "
                    "via asyncio.gather: %s",
                    [a.name for a in phase1_agents],
                )
                results: list[AgentResult] = await asyncio.gather(
                    *(agent.arun(state, self.llm) for agent in phase1_agents)
                )
                for res in results:
                    traces.append(res.trace)

            # 3. Phase 2: Synthesis (EA) & Quality Gate Refinement Loop (VA)
            if self.ea is None:
                logger.info(
                    "No ExplainerAgent configured; synthesizing deterministic summary."
                )
                final_narrative = self._build_deterministic_fallback(
                    state, reason="ExplainerAgent (EA) não configurado na camada."
                )
                fallback_used = True
            else:
                effective_max_attempts = (
                    self.max_attempts if self.va is not None else 1
                )

                for attempt in range(1, effective_max_attempts + 1):
                    attempts_used = attempt
                    logger.info(
                        "Running ExplainerAgent synthesis (attempt %d/%d).",
                        attempt,
                        effective_max_attempts,
                    )
                    ea_result = await self.ea.arun(state, self.llm)
                    traces.append(ea_result.trace)
                    current_narrative = ea_result.output
                    final_narrative = current_narrative

                    if self.va is None:
                        break

                    logger.info(
                        "Running ValidatorAgent quality gate (attempt %d/%d).",
                        attempt,
                        effective_max_attempts,
                    )
                    va_result = await self.va.arun(state, self.llm)
                    traces.append(va_result.trace)

                    if isinstance(va_result, ValidationResult):
                        validation_score = va_result.score
                        validation_passed = va_result.passed
                        validation_critique = va_result.critique

                        # Track best candidate narrative observed across attempts
                        if va_result.score > best_score:
                            best_score = va_result.score
                            best_narrative = current_narrative

                        if va_result.parse_error:
                            logger.error(
                                "ValidatorAgent encountered a JSON parse error on "
                                "attempt %d. Activating deterministic fail-safe.",
                                attempt,
                            )
                            fallback_used = True
                            validation_passed = False
                            final_narrative = self._build_deterministic_fallback(
                                state,
                                reason=(
                                    "Falha de formato (parse_error) na auditoria "
                                    "do ValidatorAgent."
                                ),
                            )
                            traces.append(
                                AgentStepTrace(
                                    agent_name="orchestrator",
                                    execution_time_ms=0.0,
                                    summary=(
                                        "Fail-safe fallback triggered due to "
                                        "validator JSON parse error."
                                    ),
                                    status="fallback",
                                    details={"attempt": attempt},
                                )
                            )
                            break

                        if va_result.passed:
                            logger.info(
                                "Explanation approved by ValidatorAgent on attempt %d "
                                "(score=%.2f >= threshold=%.2f).",
                                attempt,
                                va_result.score,
                                self.va.threshold,
                            )
                            break

                        logger.warning(
                            "ValidatorAgent rejected draft on attempt %d/%d "
                            "(score=%.2f < threshold=%.2f). Critique: %s",
                            attempt,
                            effective_max_attempts,
                            va_result.score,
                            self.va.threshold,
                            va_result.critique,
                        )

                # Check if refinement loop exhausted without approval
                if (
                    self.va is not None
                    and validation_passed is False
                    and not fallback_used
                ):
                    logger.warning(
                        "Refinement loop exhausted after %d attempts (best_score=%.2f, "
                        "last_score=%.2f). Delivering best candidate narrative with "
                        "quality warning.",
                        attempts_used,
                        best_score if best_score >= 0.0 else 0.0,
                        validation_score or 0.0,
                    )
                    fallback_used = True
                    # Keep best candidate narrative and align score with delivered text
                    if best_narrative:
                        final_narrative = best_narrative
                    if best_score >= 0.0:
                        validation_score = best_score

                    summary_msg = (
                        f"Quality gate threshold not met after {attempts_used} "
                        f"attempts (delivered best_score={validation_score}). "
                        f"Delivered best draft with low-confidence indicator."
                    )
                    traces.append(
                        AgentStepTrace(
                            agent_name="orchestrator",
                            execution_time_ms=0.0,
                            summary=summary_msg,
                            status="fallback",
                            details={
                                "attempts_used": attempts_used,
                                "best_score": best_score,
                                "last_score": va_result.score
                                if isinstance(va_result, ValidationResult)
                                else None,
                            },
                        )
                    )

        except Exception as exc:
            # 4. Unplugged Fail-Safe Mode on any unexpected LLM / tool / runtime failure
            elapsed_err_ms = (time.perf_counter() - pipeline_start) * 1000.0
            logger.error(
                "Orchestrator caught exception (%s: %s). Activating Unplugged "
                "fail-safe mode.",
                type(exc).__name__,
                exc,
            )
            fallback_used = True
            if self.va is not None:
                validation_passed = False
            final_narrative = self._build_deterministic_fallback(
                state,
                reason=f"Exceção interceptada ({type(exc).__name__}: {exc})",
            )
            traces.append(
                AgentStepTrace(
                    agent_name="orchestrator",
                    execution_time_ms=elapsed_err_ms,
                    summary=(
                        f"Unplugged fail-safe activated due to "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    status="fallback",
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )

        # 5. Persist conversation turn if explicit session_id + session_store
        if explicit_session and self.session_store is not None:
            try:
                user_msg = ChatMessage(
                    role=ROLE_USER,
                    content=user_query,
                    metadata={"session_id": resolved_session_id},
                )
                assistant_msg = ChatMessage(
                    role=ROLE_ASSISTANT,
                    content=final_narrative,
                    metadata={
                        "session_id": resolved_session_id,
                        "validation_score": validation_score,
                        "validation_passed": validation_passed,
                        "attempts_used": attempts_used,
                        "fallback_used": fallback_used,
                    },
                )
                await self.session_store.aappend_message(
                    resolved_session_id, user_msg
                )
                await self.session_store.aappend_message(
                    resolved_session_id, assistant_msg
                )
                logger.debug(
                    "Persisted user and assistant messages to session '%s'.",
                    resolved_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist messages for session '%s': %s",
                    resolved_session_id,
                    exc,
                )

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        logger.info(
            "Orchestration completed in %.2fms (attempts=%d, fallback=%s, score=%s).",
            total_ms,
            attempts_used,
            fallback_used,
            validation_score,
        )

        return ExplanationResponse(
            narrative=final_narrative,
            validation_score=validation_score,
            validation_passed=validation_passed,
            validation_critique=validation_critique,
            attempts_used=attempts_used,
            fallback_used=fallback_used,
            session_id=resolved_session_id,
            execution_trace=traces,
        )
