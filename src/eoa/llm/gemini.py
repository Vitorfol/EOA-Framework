"""Google Gemini LLM provider for the EOA framework.

Supports Mode 1 (API key + model name) and Mode 2 (Bring Your Own genai.Client)
using the official modern `google-genai` SDK.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

from eoa.llm.base import LLM

# Try importing google-genai gracefully to keep core ultra-lightweight
try:
    from google import genai
    from google.genai import types

    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False


class GeminiProvider(LLM):
    """Google Gemini model provider using the official `google-genai` SDK.

    Provides synchronous (`generate`) and non-blocking asynchronous
    (`agenerate`) completions, supporting system instructions, model selection,
    and automatic retrieval of credentials from `GEMINI_API_KEY`.

    Attributes:
        model: Target model name (e.g., 'gemini-2.5-flash', 'gemini-3.8-flash').
        client: The underlying Google GenAI client instance.

    Example:
        >>> # Mode 1: Using environment variable GEMINI_API_KEY
        >>> provider = GeminiProvider(model="gemini-2.5-flash")
        >>> response = provider.generate("Summarize hydraulic status.")

        >>> # Mode 1: Explicit API Key
        >>> provider = GeminiProvider(api_key="AIzaSy...", model="gemini-3.8-flash")

        >>> # Mode 2: Existing configured client (with proxy, custom TLS, etc.)
        >>> my_client = genai.Client(api_key="...", http_options={"proxy": "..."})
        >>> provider = GeminiProvider(client=my_client)
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        **client_options: Any,
    ) -> None:
        """Initialize the GeminiProvider.

        Args:
            api_key: Optional Gemini API key. If omitted, checks
                `GEMINI_API_KEY` environment variable.
            model: Optional model identifier. Defaults to `GEMINI_MODEL` env var
                or 'gemini-3.5-flash-lite'.
            client: Optional pre-configured `genai.Client` instance (Mode 2).
            **client_options: Additional options forwarded to `genai.Client`
                when created.

        Raises:
            ImportError: If the `google-genai` package is not installed.
            ValueError: If neither a client nor an API key can be found.
        """
        if not _GENAI_AVAILABLE:
            msg = (
                "The 'google-genai' package is required to use GeminiProvider. "
                "Install it using: pip install 'eoa[gemini]'"
            )
            raise ImportError(msg)

        resolved_model = (
            model
            or os.environ.get("GEMINI_MODEL")
            or self.DEFAULT_MODEL
        )
        self.model = resolved_model

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not resolved_key:
                msg = (
                    "No Gemini API key provided. Either pass 'api_key=...', "
                    "set the 'GEMINI_API_KEY' environment variable, or pass "
                    "an existing client instance via 'client=genai.Client(...)'."
                )
                raise ValueError(msg)

            self.client = genai.Client(api_key=resolved_key, **client_options)

    def _build_config(
        self,
        system_prompt: str | None,
        kwargs: dict[str, Any],
    ) -> types.GenerateContentConfig | None:
        """Construct a GenerateContentConfig from system prompt and kwargs."""
        config_args: dict[str, Any] = {}

        if system_prompt:
            config_args["system_instruction"] = system_prompt

        # Extract standard generation hyperparameters if present
        if "temperature" in kwargs:
            config_args["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            config_args["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            config_args["top_k"] = kwargs.pop("top_k")
        if "max_output_tokens" in kwargs:
            config_args["max_output_tokens"] = kwargs.pop("max_output_tokens")
        if "stop_sequences" in kwargs:
            config_args["stop_sequences"] = kwargs.pop("stop_sequences")

        # Allow passing pre-built config directly
        if "config" in kwargs:
            return kwargs.pop("config")  # type: ignore[no-any-return]

        if not config_args:
            return None

        return types.GenerateContentConfig(**config_args)

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """Detect transient server or rate limit errors (429, 503) for retry."""
        if hasattr(exc, "code") and exc.code in (429, 500, 502, 503, 504):
            return True
        if hasattr(exc, "status_code") and exc.status_code in (
            429,
            500,
            502,
            503,
            504,
        ):
            return True
        err_str = str(exc).upper()
        return (
            "429" in err_str
            or "503" in err_str
            or "UNAVAILABLE" in err_str
            or "RESOURCE_EXHAUSTED" in err_str
            or "RATE LIMIT" in err_str
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate response using Google Gemini.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional guidance or domain instructions.
            **kwargs: Extra parameters (temperature, max_output_tokens, etc.).

        Returns:
            The text completion returned by Gemini.
        """
        config = self._build_config(system_prompt, kwargs)
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                    **kwargs,
                )
                return response.text or ""
            except Exception as exc:
                if self._is_transient_error(exc) and attempt < max_retries - 1:
                    sleep_time = backoff + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue
                raise

        return ""

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate response using Google Gemini.

        Args:
            prompt: Main instruction or user input string.
            system_prompt: Optional guidance or domain instructions.
            **kwargs: Extra parameters (temperature, max_output_tokens, etc.).

        Returns:
            The text completion returned by Gemini.
        """
        config = self._build_config(system_prompt, kwargs)
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                    **kwargs,
                )
                return response.text or ""
            except Exception as exc:
                if self._is_transient_error(exc) and attempt < max_retries - 1:
                    sleep_time = backoff + random.uniform(0.1, 0.5)
                    await asyncio.sleep(sleep_time)
                    backoff *= 2.0
                    continue
                raise

        return ""
