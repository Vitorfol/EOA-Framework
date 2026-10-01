"""Global pytest configuration and shared fixtures for EOA test suite."""

import pytest

from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def mock_llm() -> MockLLM:
    """Fixture providing a clean, fresh MockLLM instance for each test.

    Returns:
        MockLLM initialized with default response.
    """
    return MockLLM(default_response="Default mock explanation response.")
