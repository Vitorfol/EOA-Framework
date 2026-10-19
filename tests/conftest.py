"""Global pytest configuration and shared fixtures for EOA test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.mocks.mock_llm import MockLLM


def _load_env_file() -> None:
    """Auto-load .env file from project root into os.environ if present."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        try:
            content = env_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            pass


_load_env_file()


@pytest.fixture
def mock_llm() -> MockLLM:
    """Fixture providing a clean, fresh MockLLM instance for each test.

    Returns:
        MockLLM initialized with default response.
    """
    return MockLLM(default_response="Default mock explanation response.")
