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
from eoa.llm.callable import CallableLLMAdapter
from eoa.llm.factory import create_llm
from eoa.llm.gemini import GeminiProvider
from eoa.memory import (
    ChatMessage,
    InMemorySessionStore,
    SessionStore,
    SQLiteSessionStore,
)
from eoa.prompts import PromptManager

__version__: str = "0.1.0"

__all__: tuple[str, ...] = (
    "__version__",
    "PredictionArtifact",
    "XAIArtifact",
    "DomainContext",
    "AgentStepTrace",
    "ExplanationResponse",
    "LLM",
    "CallableLLMAdapter",
    "GeminiProvider",
    "create_llm",
    "ChatMessage",
    "InMemorySessionStore",
    "SessionStore",
    "SQLiteSessionStore",
    "PromptManager",
)
