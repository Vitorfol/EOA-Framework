"""Callable adapter enabling any Python function to operate as an EOA LLM.

Supports Mode 3 (Custom Functions): local models (Ollama, vLLM, HuggingFace),
third-party providers (OpenAI, Anthropic, Mistral), or custom enterprise gateways.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eoa.llm.base import LLM


class CallableLLMAdapter(LLM):
    """Adapts an arbitrary Python callable into the EOA LLM protocol.

    Seamlessly supports both synchronous and asynchronous callables,
    with flexible parameter inspection (single prompt argument, dual
    prompt + system_prompt, or arbitrary keyword arguments).

    Attributes:
        func: The underlying callable to invoke.

    Example:
        >>> # Sync function (e.g., custom model or simple rule engine)
        >>> def my_local_llm(prompt: str) -> str:
        ...     return f"Processed: {prompt}"
        >>> adapter = CallableLLMAdapter(my_local_llm)
        >>> adapter.generate("Hello")
        'Processed: Hello'

        >>> # Third-party client wrapper (e.g., OpenAI or Ollama)
        >>> def openai_wrapper(prompt: str, system_prompt: str | None = None) -> str:
        ...     return client.chat.completions.create(...).choices[0].message.content
        >>> llm = CallableLLMAdapter(openai_wrapper)
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize the adapter with a callable.

        Args:
            func: A synchronous or asynchronous callable that takes at least
                a prompt string and returns text.

        Raises:
            TypeError: If the provided argument is not callable.
        """
        if not callable(func):
            msg = (
                "CallableLLMAdapter expects a callable object "
                f"(function, lambda, or method), got {type(func).__name__}."
            )
            raise TypeError(msg)

        self.func = func
        self._is_coroutine_fn = inspect.iscoroutinefunction(func)
        self._sig = inspect.signature(func)
        self._accepts_system_prompt = self._check_accepts_param("system_prompt")
        self._accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in self._sig.parameters.values()
        )

    def _check_accepts_param(self, name: str) -> bool:
        """Check if underlying callable explicitly declares or accepts parameter."""
        params = self._sig.parameters
        if name in params:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in params.values()
        )

    def _prepare_call_args(
        self,
        prompt: str,
        system_prompt: str | None,
        kwargs: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Filter and map arguments matching the callable's signature."""
        positional_args: list[Any] = [prompt]
        call_kwargs: dict[str, Any] = {}

        if self._accepts_system_prompt and system_prompt is not None:
            call_kwargs["system_prompt"] = system_prompt

        if self._accepts_kwargs:
            call_kwargs.update(kwargs)
        else:
            for k, v in kwargs.items():
                if k in self._sig.parameters:
                    call_kwargs[k] = v

        return positional_args, call_kwargs

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously execute the callable.

        If the callable is asynchronous, runs it safely without blocking
        or corrupting external event loops.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional guidance or persona instructions.
            **kwargs: Extra parameters forwarded to the callable if accepted.

        Returns:
            The string response produced by the callable.
        """
        args, call_kwargs = self._prepare_call_args(prompt, system_prompt, kwargs)

        if not self._is_coroutine_fn:
            result = self.func(*args, **call_kwargs)
            return str(result)

        # Handling async callable called in synchronous context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If event loop is already active, run in thread to avoid deadlock
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self.func(*args, **call_kwargs)
                )
                return str(future.result())

        return str(asyncio.run(self.func(*args, **call_kwargs)))

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously execute the callable.

        If the callable is synchronous, offloads it to a thread worker
        via asyncio.to_thread to keep the event loop non-blocking.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional guidance or persona instructions.
            **kwargs: Extra parameters forwarded to the callable if accepted.

        Returns:
            The string response produced by the callable.
        """
        args, call_kwargs = self._prepare_call_args(prompt, system_prompt, kwargs)

        if self._is_coroutine_fn:
            result = await self.func(*args, **call_kwargs)
            return str(result)

        # Non-blocking execution for sync callable in async loop
        result = await asyncio.to_thread(self.func, *args, **call_kwargs)
        return str(result)
