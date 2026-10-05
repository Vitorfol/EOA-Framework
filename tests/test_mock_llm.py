"""Unit tests for MockLLM and package structure verification."""

import pytest

import eoa
from eoa.llm.base import LLM
from tests.mocks.mock_llm import MockLLM


def test_package_metadata() -> None:
    """Verify package importability and version definition."""
    assert hasattr(eoa, "__version__")
    assert eoa.__version__ == "0.1.0"


def test_mock_llm_implements_protocol() -> None:
    """Ensure MockLLM implements the runtime-checkable LLM protocol."""
    mock = MockLLM()
    assert isinstance(mock, LLM)


def test_mock_llm_default_response() -> None:
    """Test standard synchronous response generation with default response."""
    mock = MockLLM(default_response="Healthy pump.")
    response = mock.generate("Explain anomaly")

    assert response == "Healthy pump."
    assert mock.call_count == 1
    assert mock.last_prompt == "Explain anomaly"
    assert mock.last_system_prompt is None
    assert mock.last_call is not None
    assert not mock.last_call.is_async


@pytest.mark.asyncio
async def test_mock_llm_async_agenerate() -> None:
    """Test asynchronous response generation with call inspection."""
    mock = MockLLM(default_response="Async explanation.")
    response = await mock.agenerate(
        prompt="Inspect hydraulic pressure",
        system_prompt="You are a domain expert",
        temperature=0.2,
    )

    assert response == "Async explanation."
    assert mock.call_count == 1
    assert mock.last_prompt == "Inspect hydraulic pressure"
    assert mock.last_system_prompt == "You are a domain expert"
    assert mock.last_call is not None
    assert mock.last_call.is_async
    assert mock.last_call.kwargs.get("temperature") == 0.2


def test_mock_llm_response_queue() -> None:
    """Test FIFO sequence of queued responses."""
    mock = MockLLM()
    mock.queue_responses("First answer", "Second answer")

    r1 = mock.generate("Query 1")
    r2 = mock.generate("Query 2")
    r3 = mock.generate("Query 3")  # falls back to default

    assert r1 == "First answer"
    assert r2 == "Second answer"
    assert r3 == "Mocked LLM response"
    assert mock.call_count == 3


def test_mock_llm_canned_responses() -> None:
    """Test keyword-based canned responses."""
    mock = MockLLM(
        default_response="Default answer",
        canned_responses={
            "pressure": "Pressure valve malfunction detected.",
            "temperature": "Thermal overheating in cooler unit.",
        },
    )

    r_pressure = mock.generate("What happened to the pressure sensor PS1?")
    r_temp = mock.generate("Check temperature TS1")
    r_other = mock.generate("Random unrelated query")

    assert r_pressure == "Pressure valve malfunction detected."
    assert r_temp == "Thermal overheating in cooler unit."
    assert r_other == "Default answer"


def test_mock_llm_custom_generator() -> None:
    """Test custom generator callable."""
    mock = MockLLM(custom_generator=lambda prompt, sys: f"ECHO: {prompt} | SYS: {sys}")

    res = mock.generate("Test prompt", system_prompt="Sys prompt")
    assert res == "ECHO: Test prompt | SYS: Sys prompt"


def test_mock_llm_raise_error_simulation() -> None:
    """Test simulating network failures or timeouts for fail-safe testing."""
    mock = MockLLM(raise_error=TimeoutError("LLM service timeout"))

    with pytest.raises(TimeoutError, match="LLM service timeout"):
        mock.generate("Prompt that should fail")

    # Clear error and verify recovery
    mock.set_error(None)
    mock.set_response("Recovered response")
    assert mock.generate("Prompt after recovery") == "Recovered response"


def test_mock_llm_reset() -> None:
    """Test resetting mock history and queued responses."""
    mock = MockLLM()
    mock.queue_responses("Answer 1", "Answer 2")
    mock.generate("Call 1")

    assert mock.call_count == 1
    mock.reset()
    assert mock.call_count == 0
    assert mock.last_call is None
    assert mock.calls == []

    # Queue should be cleared, returning default response
    assert mock.generate("After reset") == "Mocked LLM response"


def test_conftest_mock_fixture(mock_llm: MockLLM) -> None:
    """Verify that the mock_llm pytest fixture operates correctly."""
    assert isinstance(mock_llm, LLM)
    assert mock_llm.generate("Hello") == "Default mock explanation response."
    assert mock_llm.call_count == 1
