"""Deterministic MockLLM for unit testing and offline development.

Allows simulating LLM completions, canned responses, sequential answers,
and network/provider failures without API keys or token expenditure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eoa.llm.base import LLM


@dataclass
class LLMCallRecord:
    """Record of an invocation to the mock LLM provider.

    Attributes:
        prompt: The main user/instruction prompt sent to the LLM.
        system_prompt: The system prompt or instructions provided, if any.
        kwargs: Extra keyword arguments passed to the call.
        is_async: Boolean flag indicating if call was made asynchronously.
    """

    prompt: str
    system_prompt: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)
    is_async: bool = False


class MockLLM(LLM):
    """Deterministic mock implementation of the LLM protocol.

    Useful for zero-cost, deterministic, and offline testing of EOA agents,
    orchestrator refinement loops, and error-handling mechanisms.

    Example:
        >>> mock = MockLLM(default_response="System is healthy.")
        >>> response = mock.generate("Check hydraulic pump status")
        >>> assert response == "System is healthy."
        >>> assert mock.call_count == 1
        >>> assert mock.last_prompt == "Check hydraulic pump status"
    """

    def __init__(
        self,
        default_response: str = "Mocked LLM response",
        responses: list[str] | None = None,
        canned_responses: dict[str, str] | None = None,
        custom_generator: Callable[[str, str | None], str] | None = None,
        raise_error: Exception | None = None,
        simulated_latency: float = 0.0,
    ) -> None:
        """Initialize the MockLLM.

        Args:
            default_response: Fallback text returned when no other match occurs.
            responses: Optional list/queue of responses returned sequentially.
            canned_responses: Optional mapping of keyword/substring to response.
            custom_generator: Optional custom callable `(prompt, system_prompt) -> str`.
            raise_error: Exception to raise when generate/agenerate is called.
            simulated_latency: Latency in seconds to simulate network delay.
        """
        self.default_response = default_response
        self._response_queue: list[str] = list(responses) if responses else []
        self.canned_responses = canned_responses or {}
        self.custom_generator = custom_generator
        self.raise_error = raise_error
        self.simulated_latency = simulated_latency
        self._calls: list[LLMCallRecord] = []

    @property
    def calls(self) -> list[LLMCallRecord]:
        """Return the list of all recorded calls."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Return the number of times this mock has been invoked."""
        return len(self._calls)

    @property
    def last_call(self) -> LLMCallRecord | None:
        """Return the most recent call record, or None if never invoked."""
        return self._calls[-1] if self._calls else None

    @property
    def last_prompt(self) -> str | None:
        """Return the prompt from the most recent call, or None."""
        return self._calls[-1].prompt if self._calls else None

    @property
    def last_system_prompt(self) -> str | None:
        """Return the system prompt from the most recent call, or None."""
        return self._calls[-1].system_prompt if self._calls else None

    def set_response(self, response: str) -> None:
        """Set a single default response for subsequent calls.

        Args:
            response: The response string to return.
        """
        self.default_response = response

    def queue_responses(self, *responses: str) -> None:
        """Enqueue multiple responses to be returned in FIFO sequence.

        Args:
            *responses: Responses to be returned in sequential calls.
        """
        self._response_queue.extend(responses)

    def set_error(self, error: Exception | None) -> None:
        """Configure an exception to simulate failure or pass None to clear it.

        Args:
            error: The exception instance to raise, or None to resume normal responses.
        """
        self.raise_error = error

    def reset(self) -> None:
        """Reset the recorded calls history and clear remaining queued responses."""
        self._calls.clear()
        self._response_queue.clear()

    def _resolve_response(self, prompt: str, system_prompt: str | None) -> str:
        """Internal helper to determine the response according to precedence rules."""
        if self.raise_error is not None:
            raise self.raise_error

        # 1. Custom generator takes precedence if defined
        if self.custom_generator is not None:
            return self.custom_generator(prompt, system_prompt)

        # 2. Sequential queue
        if self._response_queue:
            return self._response_queue.pop(0)

        # 3. Substring/keyword matching
        for key, canned in self.canned_responses.items():
            if key in prompt or (system_prompt and key in system_prompt):
                return canned

        # 4. Fallback default
        return self.default_response

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate a mock LLM completion.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional system persona or guidance instructions.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Configured mock completion text.

        Raises:
            Exception: If `raise_error` is configured.
        """
        self._calls.append(
            LLMCallRecord(
                prompt=prompt,
                system_prompt=system_prompt,
                kwargs=kwargs,
                is_async=False,
            )
        )
        if self.simulated_latency > 0:
            import time

            time.sleep(self.simulated_latency)

        return self._resolve_response(prompt, system_prompt)

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate a mock LLM completion.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional system persona or guidance instructions.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Configured mock completion text.

        Raises:
            Exception: If `raise_error` is configured.
        """
        self._calls.append(
            LLMCallRecord(
                prompt=prompt,
                system_prompt=system_prompt,
                kwargs=kwargs,
                is_async=True,
            )
        )
        if self.simulated_latency > 0:
            await asyncio.sleep(self.simulated_latency)

        return self._resolve_response(prompt, system_prompt)
