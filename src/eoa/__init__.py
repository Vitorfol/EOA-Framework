"""EOA - Explanation-Oriented Agents Framework.

A modular, lightweight Python framework for explainable AI in mission-critical
and predictive maintenance systems, featuring multi-agent quality gates
and unplugged fail-safe operation.
"""

from eoa.contracts import (
    AgentStepTrace,
    DomainContext,
    ExplanationResponse,
    PredictionArtifact,
    XAIArtifact,
)
from eoa.llm.base import LLM

__version__: str = "0.1.0"

__all__: tuple[str, ...] = (
    "__version__",
    "PredictionArtifact",
    "XAIArtifact",
    "DomainContext",
    "AgentStepTrace",
    "ExplanationResponse",
    "LLM",
)
