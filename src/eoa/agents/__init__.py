"""Agent modules and base abstractions for EOA (CMA, AA, EA, VA, Specialized)."""

from eoa.agents.artifacts import ArtifactsAgent
from eoa.agents.base import (
    AgentResult,
    AgentState,
    BaseAgent,
    ValidationResult,
)
from eoa.agents.context import (
    ContextManagerAgent,
    ContextRetriever,
    StructuredContextRetriever,
)
from eoa.agents.explainer import ExplainerAgent
from eoa.agents.specialized import SpecializedAgent
from eoa.agents.validator import ValidatorAgent

__all__: tuple[str, ...] = (
    "AgentState",
    "AgentResult",
    "ValidationResult",
    "BaseAgent",
    "ContextRetriever",
    "StructuredContextRetriever",
    "ContextManagerAgent",
    "ArtifactsAgent",
    "ExplainerAgent",
    "ValidatorAgent",
    "SpecializedAgent",
)
