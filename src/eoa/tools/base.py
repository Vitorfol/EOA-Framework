"""Tool abstraction and execution engine for EOA agents."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_TYPE_NAME_MAP: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
    "list": "array",
    "dict": "object",
}


class Tool:
    """Encapsulates a callable function as an inspectable agent tool.

    Extracts parameter names, descriptions, JSON schemas, and signatures
    automatically, supporting both synchronous and asynchronous execution.

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
            p_name
            for p_name, param in self.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        self.parameters_schema: dict[str, Any] = self._build_parameters_schema()

    @staticmethod
    def _resolve_json_type(annotation: Any) -> str:
        """Map a Python type annotation to a JSON Schema primitive type."""
        if annotation is inspect.Parameter.empty:
            return "any"
        if isinstance(annotation, type):
            return _TYPE_NAME_MAP.get(annotation.__name__, "any")
        ann_str = str(annotation).strip()
        return _TYPE_NAME_MAP.get(ann_str, "any")

    def _build_parameters_schema(self) -> dict[str, Any]:
        """Build a JSON-Schema-like dictionary describing tool parameters."""
        properties: dict[str, Any] = {}
        for p_name, param in self.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            prop_info: dict[str, Any] = {
                "type": self._resolve_json_type(param.annotation),
            }
            if param.default is not inspect.Parameter.empty:
                prop_info["default"] = param.default
            properties[p_name] = prop_info

        return {
            "type": "object",
            "properties": properties,
            "required": list(self._required_parameters),
        }

    def to_descriptor(self) -> dict[str, Any]:
        """Return a serializable descriptor of this tool for LLM reasoning."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

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

    def _coerce_value(self, param_name: str, value: Any) -> Any:
        """Safely coerce extracted string/numeric values to match type annotations."""
        param = self.parameters.get(param_name)
        if param is None or value is None:
            return value
        json_type = self._resolve_json_type(param.annotation)
        try:
            if json_type == "integer" and not isinstance(value, bool):
                return int(value)
            if json_type == "number" and not isinstance(value, bool):
                return float(value)
            if json_type == "string" and not isinstance(value, str):
                return str(value)
        except (ValueError, TypeError):
            pass
        return value

    def _filter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only kwargs accepted by the tool signature, unless taking **kwargs."""
        if self.has_var_keyword:
            return {k: self._coerce_value(k, v) for k, v in kwargs.items()}
        return {
            k: self._coerce_value(k, v)
            for k, v in kwargs.items()
            if k in self.parameters
        }

    def __repr__(self) -> str:
        params_str = ", ".join(self.parameters.keys())
        return f"Tool(name='{self.name}', params=[{params_str}], async={self.is_async})"
