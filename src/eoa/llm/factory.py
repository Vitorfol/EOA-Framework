"""Factory function and resolution utilities for EOA LLM providers.

Provides developer-friendly unified entry point (`create_llm`)
supporting all 3 connection modes transparently:
- Mode 1: Direct Credentials / Provider name
- Mode 2: Existing instantiated Client (BYO Client)
- Mode 3: Custom Python functions (Custom Callables)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from eoa.llm.base import LLM
from eoa.llm.callable import CallableLLMAdapter


def create_llm(
    llm: LLM | Callable[..., Any] | Any | None = None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLM:
    """Create or resolve an EOA-compliant LLM instance across the 3 modes.

    Resolution Flow:
        1. Idempotency: If `llm` is already an `LLM` instance, returns it directly.
        2. Mode 2 (BYO Client): If `llm` is a Google `genai.Client`, wraps it in
           `GeminiProvider(client=llm)`.
        3. Mode 3 (Custom Callable): If `llm` is a callable (sync or async),
           wraps it in `CallableLLMAdapter(llm)`.
        4. Mode 1 (Explicit Provider): If `provider='gemini'`, creates
           `GeminiProvider(api_key=..., model=...)`.
        5. Mode 1 (Zero-Config / Auto-Detect): If no args and `GEMINI_API_KEY` is set,
           creates `GeminiProvider(model=...)`.
        6. Fallback: Raises `ValueError` with guidance on the 3 connection modes.

    Args:
        llm: Existing LLM instance, callable function, or third-party client.
        provider: Provider identifier string (e.g., 'gemini').
        api_key: API key string (if not using environment variables).
        model: Model name/identifier (defaults to 'gemini-3.5-flash-lite').
        **kwargs: Additional parameters forwarded to the provider or client.

    Returns:
        An instance conforming to the `LLM` protocol.

    Raises:
        ValueError: If configuration is invalid or missing required parameters.
        TypeError: If an unsupported object type is passed as `llm`.

    Examples:
        >>> # Case 1: Idempotency (already an LLM instance)
        >>> llm = create_llm(existing_llm_instance)

        >>> # Case 2: Mode 2 (Existing client instance)
        >>> client = genai.Client(...)
        >>> llm = create_llm(client)

        >>> # Case 3: Mode 3 (Custom callable function or lambda)
        >>> llm = create_llm(lambda prompt: "Custom response")

        >>> # Case 4: Mode 1 (Explicit provider and key)
        >>> llm = create_llm(provider="gemini", api_key="AIzaSy...")

        >>> # Case 5: Mode 1 (Zero-config from GEMINI_API_KEY environment variable)
        >>> llm = create_llm()
    """
    # 1. Already an EOA-compliant LLM
    if isinstance(llm, LLM):
        return llm

    # 2. Check if it's an existing Google GenAI client (Mode 2)
    client_type_name = type(llm).__name__ if llm is not None else ""
    client_module_name = type(llm).__module__ if llm is not None else ""
    if "genai" in client_module_name and client_type_name == "Client":
        from eoa.llm.gemini import GeminiProvider

        return GeminiProvider(client=llm, model=model, **kwargs)

    # 3. Custom callable (Mode 3)
    if callable(llm):
        return CallableLLMAdapter(llm)

    # 4. Explicit provider string (Mode 1)
    normalized_provider = provider.lower().strip() if provider else None
    if normalized_provider == "gemini":
        from eoa.llm.gemini import GeminiProvider

        return GeminiProvider(api_key=api_key, model=model, **kwargs)

    if normalized_provider is not None:
        msg = (
            f"Unsupported provider '{provider}'. EOA v0.1.0 natively supports "
            "provider='gemini'. To use OpenAI, Anthropic, or local models, connect "
            "them using Mode 2 (client) or Mode 3 (custom callable function): "
            "create_llm(llm=my_function)."
        )
        raise ValueError(msg)

    # 5. Auto-detection via environment variables (Zero-config Mode 1)
    if llm is None and os.environ.get("GEMINI_API_KEY"):
        from eoa.llm.gemini import GeminiProvider

        return GeminiProvider(model=model, **kwargs)

    # 6. Fallback error with educational guidance
    msg = (
        "Could not resolve LLM configuration. Please use one of the 3 modes:\n"
        "  1. Mode 1 (Gemini): create_llm(provider='gemini') or set GEMINI_API_KEY\n"
        "  2. Mode 2 (Client): create_llm(llm=my_client)\n"
        "  3. Mode 3 (Function): create_llm(llm=my_custom_function)"
    )
    raise ValueError(msg)
