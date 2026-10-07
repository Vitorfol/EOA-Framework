"""Unit and integration tests for EOA LLM abstractions and providers (Epic 03).

Tests:
1. CallableLLMAdapter (Mode 3: sync and async user functions).
2. GeminiProvider (Mode 1 & Mode 2: unit tests with client mocks).
3. create_llm / as_llm factory resolution and educational error messages.
4. Optional live integration test with Gemini API when GEMINI_API_KEY is available.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import eoa
from eoa.llm.base import LLM
from eoa.llm.callable import CallableLLMAdapter
from eoa.llm.factory import as_llm, create_llm
from eoa.llm.gemini import GeminiProvider
from tests.mocks.mock_llm import MockLLM

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

# ==============================================================================
# 1. CallableLLMAdapter Tests (Mode 3)
# ==============================================================================


def test_callable_adapter_sync_single_arg() -> None:
    """Test sync function accepting only prompt string."""
    def simple_echo(prompt: str) -> str:
        return f"ECHO: {prompt}"

    adapter = CallableLLMAdapter(simple_echo)
    assert isinstance(adapter, LLM)

    # Sync call
    res = adapter.generate("Test prompt")
    assert res == "ECHO: Test prompt"


def test_callable_adapter_sync_with_system_prompt() -> None:
    """Test sync function accepting both prompt and system_prompt."""
    def expert_bot(prompt: str, system_prompt: str | None = None) -> str:
        return f"[{system_prompt}] -> {prompt}"

    adapter = CallableLLMAdapter(expert_bot)
    res = adapter.generate("Check pump", system_prompt="Engineer")
    assert res == "[Engineer] -> Check pump"


def test_callable_adapter_sync_with_kwargs() -> None:
    """Test sync function accepting kwargs."""
    def kwargs_bot(prompt: str, **kwargs: object) -> str:
        temp = kwargs.get("temperature", 0.0)
        return f"Prompt: {prompt} | Temp: {temp}"

    adapter = CallableLLMAdapter(kwargs_bot)
    res = adapter.generate("Pressure alert", temperature=0.7)
    assert res == "Prompt: Pressure alert | Temp: 0.7"


@pytest.mark.asyncio
async def test_callable_adapter_sync_called_via_agenerate() -> None:
    """Ensure sync function is safely offloaded via to_thread in async context."""
    def blocking_calc(prompt: str) -> str:
        return f"Computed: {prompt}"

    adapter = CallableLLMAdapter(blocking_calc)
    res = await adapter.agenerate("Async run")
    assert res == "Computed: Async run"


@pytest.mark.asyncio
async def test_callable_adapter_async_coroutine() -> None:
    """Test native async coroutine function with agenerate."""
    async def async_llm(prompt: str, system_prompt: str | None = None) -> str:
        return f"ASYNC: {prompt} (Sys: {system_prompt})"

    adapter = CallableLLMAdapter(async_llm)

    # Async execution
    res = await adapter.agenerate("Sensor PS1", system_prompt="Hydraulics")
    assert res == "ASYNC: Sensor PS1 (Sys: Hydraulics)"

    # Sync execution of async coroutine
    sync_res = adapter.generate("Sync call to async fn")
    assert "ASYNC: Sync call to async fn" in sync_res


def test_callable_adapter_rejects_non_callable() -> None:
    """Ensure TypeError is raised when passing non-callable object."""
    with pytest.raises(TypeError, match="expects a callable object"):
        CallableLLMAdapter("not_a_function")  # type: ignore[arg-type]


# ==============================================================================
# 2. GeminiProvider Unit Tests (Mocked Google Client)
# ==============================================================================


def test_gemini_provider_with_explicit_key_and_custom_model() -> None:
    """Test initialization with explicit key and custom model."""
    provider = GeminiProvider(api_key="fake_key_123", model="gemini-3.8-flash")
    assert provider.model == "gemini-3.8-flash"
    assert provider.client is not None


def test_gemini_provider_missing_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify informative error if no API key or client is provided."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Gemini API key provided"):
        GeminiProvider()


def test_gemini_provider_sync_generate_mocked() -> None:
    """Test synchronous generate calling client.models.generate_content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The cooler efficiency is degraded by 80%."
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client, model="gemini-2.5-flash")
    result = provider.generate(
        "Explain anomaly",
        system_prompt="Industrial Expert",
        temperature=0.2,
    )

    assert result == "The cooler efficiency is degraded by 80%."
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "Explain anomaly"
    assert call_kwargs["config"].system_instruction == "Industrial Expert"
    assert call_kwargs["config"].temperature == 0.2


@pytest.mark.asyncio
async def test_gemini_provider_async_agenerate_mocked() -> None:
    """Test asynchronous agenerate calling client.aio.models.generate_content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Async response from Gemini."

    # Setup async mock
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(client=mock_client, model="gemini-3.8-flash")
    result = await provider.agenerate(
        "Inspect valve",
        system_prompt="Safety Auditor",
    )

    assert result == "Async response from Gemini."
    mock_client.aio.models.generate_content.assert_called_once()
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert call_kwargs["contents"] == "Inspect valve"
    assert call_kwargs["config"].system_instruction == "Safety Auditor"


# ==============================================================================
# 3. Factory Resolution Tests (create_llm / as_llm)
# ==============================================================================


def test_create_llm_with_existing_llm_instance() -> None:
    """Passing an existing LLM instance returns it directly."""
    mock = MockLLM(default_response="Mock answer")
    resolved = create_llm(mock)
    assert resolved is mock


def test_create_llm_with_callable_mode_3() -> None:
    """Passing a function returns CallableLLMAdapter."""
    def my_fn(prompt: str) -> str:
        return prompt

    resolved = create_llm(my_fn)
    assert isinstance(resolved, CallableLLMAdapter)
    assert resolved.generate("Hello") == "Hello"


def test_create_llm_with_provider_gemini() -> None:
    """Passing provider='gemini' creates GeminiProvider."""
    resolved = create_llm(
        provider="gemini",
        api_key="test_key",
        model="gemini-2.5-flash-lite",
    )
    assert isinstance(resolved, GeminiProvider)
    assert resolved.model == "gemini-2.5-flash-lite"


def test_create_llm_with_unsupported_provider() -> None:
    """Unsupported provider strings raise an educational error."""
    with pytest.raises(ValueError, match="Unsupported provider 'openai'"):
        create_llm(provider="openai")


def test_create_llm_auto_detect_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detects Gemini when GEMINI_API_KEY environment variable is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "env_test_key_abc")
    resolved = create_llm()
    assert isinstance(resolved, GeminiProvider)


def test_create_llm_no_args_and_no_env_raises_educational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises educational error guiding the user on the 3 connection modes."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Could not resolve LLM configuration"):
        create_llm()


def test_as_llm_alias() -> None:
    """Ensure as_llm alias is identical to create_llm."""
    assert as_llm is create_llm


def test_root_package_exports_llm_components() -> None:
    """Verify all LLM components are accessible directly from root eoa."""
    assert hasattr(eoa, "LLM")
    assert hasattr(eoa, "CallableLLMAdapter")
    assert hasattr(eoa, "GeminiProvider")
    assert hasattr(eoa, "create_llm")
    assert hasattr(eoa, "as_llm")


# ==============================================================================
# 4. Live Integration Tests (Only run if GEMINI_API_KEY is exported)
# ==============================================================================


@pytest.mark.integration
def test_live_gemini_mode_1_direct_credentials() -> None:
    """Mode 1: Direct API Key with synchronous generation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Skipping live integration test: GEMINI_API_KEY is not set.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    provider = GeminiProvider(api_key=api_key, model=model_name)

    sync_resp = provider.generate(
        "Reply with exactly one word: 'MODE1_READY'.",
        temperature=0.0,
    )
    assert "MODE1_READY" in sync_resp.upper()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_gemini_mode_2_byo_client() -> None:
    """Mode 2: BYO Client with asynchronous non-blocking generation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Skipping live integration test: GEMINI_API_KEY is not set.")

    # Create native genai.Client instance explicitly (Mode 2)
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    provider = GeminiProvider(client=client, model=model_name)

    async_resp = await provider.agenerate(
        "Reply with exactly one word: 'MODE2_CONNECTED'.",
        system_prompt="You are a system health check probe.",
        temperature=0.0,
    )
    assert "MODE2_CONNECTED" in async_resp.upper()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_gemini_factory_auto_detect() -> None:
    """Mode 1 via Factory: create_llm() auto-detecting GEMINI_API_KEY from env."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Skipping live integration test: GEMINI_API_KEY is not set.")

    # Calls factory with zero args; should resolve to GeminiProvider reading env
    llm = create_llm()
    assert isinstance(llm, GeminiProvider)

    resp = await llm.agenerate(
        "Reply with exactly one word: 'FACTORY_OK'.",
        temperature=0.0,
    )
    assert "FACTORY_OK" in resp.upper()
