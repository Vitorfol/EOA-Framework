"""LLM connectivity and model abstractions for EOA."""

from eoa.llm.base import LLM
from eoa.llm.callable import CallableLLMAdapter
from eoa.llm.factory import as_llm, create_llm
from eoa.llm.gemini import GeminiProvider

__all__ = [
    "LLM",
    "CallableLLMAdapter",
    "GeminiProvider",
    "create_llm",
    "as_llm",
]
