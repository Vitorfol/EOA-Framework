"""Tool abstraction and execution engine for EOA agents."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class Tool:
    """Encapsulates a callable function as an inspectable agent tool.

    Extracts parameter names, descriptions, and signatures automatically,
    supporting both synchronous and asynchronous execution.

    Args:
        fn: The callable Python function to execute.
        name: Optional custom name. Defaults to the function's ``__name__``.
        description: Optional description. Defaults to the function's docstring.

    Example:
        >>> def fetch_user(user_id: int) -> dict:
        ...     '''Fetch user telemetry data.'''
        ...     return {"user_id": user_id, "status": "active"}
        >>> tool = Tool(fetch_user)
        >>> tool.name
        'fetch_user'
        >>> tool.can_execute_with({"user_id": 42})
        True
        >>> tool.execute(user_id=42)
        {'user_id': 42, 'status': 'active'}
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"Tool fn must be callable, got {type(fn).__name__}")

        self.fn = fn
        self.name: str = name or getattr(fn, "__name__", "unnamed_tool")
        self.description: str = (
            description or inspect.getdoc(fn) or f"Execute {self.name}."
        )

        sig = inspect.signature(fn)
        self.parameters: dict[str, inspect.Parameter] = dict(sig.parameters)
        self.is_async: bool = inspect.iscoroutinefunction(fn)
        self.has_var_keyword: bool = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in self.parameters.values()
        )
        self._required_parameters: list[str] = [
            name
            for name, param in self.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    @property
    def required_parameters(self) -> list[str]:
        """Names of parameters that do not have default values."""
        return self._required_parameters

    def can_execute_with(self, params: dict[str, Any]) -> bool:
        """Check if all required parameters are present in the provided dictionary.

        Args:
            params: Dictionary of parameters to check against tool requirements.

        Returns:
            True if all required parameters are in params, False otherwise.
        """
        required = self._required_parameters
        if not required:
            # If function takes parameters with defaults and at least one matches
            return any(k in self.parameters for k in params)
        return all(k in params for k in required)

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously with filtered kwargs.

        Args:
            **kwargs: Arguments to pass to the underlying function.

        Returns:
            Result returned by the callable.
        """
        filtered_kwargs = self._filter_kwargs(kwargs)
        if self.is_async:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self.fn(**filtered_kwargs))
                    return future.result()

            return asyncio.run(self.fn(**filtered_kwargs))
        return self.fn(**filtered_kwargs)

    async def aexecute(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with filtered kwargs.

        Args:
            **kwargs: Arguments to pass to the underlying function.

        Returns:
            Result returned by the callable.
        """
        filtered_kwargs = self._filter_kwargs(kwargs)
        if self.is_async:
            return await self.fn(**filtered_kwargs)
        return await asyncio.to_thread(self.fn, **filtered_kwargs)

    def _filter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only kwargs accepted by the tool signature, unless taking **kwargs."""
        if self.has_var_keyword:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in self.parameters}

    def __repr__(self) -> str:
        params_str = ", ".join(self.parameters.keys())
        return f"Tool(name='{self.name}', params=[{params_str}], async={self.is_async})"
