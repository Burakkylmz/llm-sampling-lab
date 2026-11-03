
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.providers.factory import (
    build_provider_registry,
    create_provider,
)
from llm_sampling_lab.providers.groq_provider import GroqProvider


def test_registry_contains_groq():
    registry = build_provider_registry()

    assert registry.available() == [
        "groq",
        "ollama",
        "openai",
    ]


def test_create_groq_provider():
    client = MagicMock()

    provider = create_provider(
        "groq",
        groq_client=client,
    )

    assert isinstance(provider, GroqProvider)


def test_unknown_provider():
    with pytest.raises(
        ValueError,
        match="Unknown provider",
    ):
        create_provider("unknown")
