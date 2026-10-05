"""Base protocols and interfaces for LLMs in EOA.

Follows PEP 544 structural subtyping (protocols) to allow zero-bloat,
pluggable language model backends.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Protocol defining the interface for any LLM in EOA.

    Any model backend (Gemini, OpenAI, local Ollama/vLLM, custom functions,
    or test mocks) must implement this interface to be utilized by EOA agents
    and orchestrators.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate a text response for the given prompt.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional system persona or guidance instructions.
            **kwargs: Additional model-specific parameters (e.g., temperature).

        Returns:
            The raw text completion from the language model.
        """
        ...

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate a text response for the given prompt.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional system persona or guidance instructions.
            **kwargs: Additional model-specific parameters (e.g., temperature).

        Returns:
            The raw text completion from the language model.
        """
        ...
