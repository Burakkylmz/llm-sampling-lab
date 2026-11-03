
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import GenerationRequest
from llm_sampling_lab.providers.capabilities import SupportStatus
from llm_sampling_lab.providers.factory import (
    build_provider_registry,
    create_provider,
)
from llm_sampling_lab.providers.openai_provider import OpenAIProvider


MODEL = "gpt-4.1-mini"


@pytest.fixture
def mock_client():
    client = MagicMock()

    completion = client.chat.completions.create.return_value
    completion.choices[0].message.content = "Hello from OpenAI"
    completion.choices[0].finish_reason = "stop"
    completion.usage.prompt_tokens = 15
    completion.usage.completion_tokens = 8

    return client


def test_model_capabilities():
    provider = OpenAIProvider()

    caps = provider.get_capabilities(MODEL)

    assert caps.temperature == SupportStatus.SUPPORTED
    assert caps.top_p == SupportStatus.SUPPORTED
    assert caps.top_k == SupportStatus.UNSUPPORTED


def test_unknown_model():
    provider = OpenAIProvider()

    caps = provider.get_capabilities("unknown-model")

    assert caps.temperature == SupportStatus.UNKNOWN


def test_generation(mock_client):
    provider = OpenAIProvider(client=mock_client)

    result = provider.generate(
        GenerationRequest(
            model=MODEL,
            prompt="Say hello",
            temperature=0.7,
            top_p=0.9,
        )
    )

    assert result.provider == "openai"
    assert result.text == "Hello from OpenAI"
    assert result.input_tokens == 15
    assert result.output_tokens == 8
    assert result.finish_reason == "stop"

    kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert kwargs["model"] == MODEL
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 256


def test_optional_sampling_parameters(mock_client):
    provider = OpenAIProvider(client=mock_client)

    provider.generate(
        GenerationRequest(
            model=MODEL,
            prompt="Hello",
        )
    )

    kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_unsupported_top_k(mock_client):
    provider = OpenAIProvider(client=mock_client)

    with pytest.raises(ValueError, match="top_k"):
        provider.generate(
            GenerationRequest(
                model=MODEL,
                prompt="Hello",
                top_k=5,
            )
        )

    mock_client.chat.completions.create.assert_not_called()


def test_unknown_model_parameter_rejected(mock_client):
    provider = OpenAIProvider(client=mock_client)

    with pytest.raises(ValueError, match="unknown"):
        provider.generate(
            GenerationRequest(
                model="unknown-model",
                prompt="Hello",
                temperature=0.7,
            )
        )

    mock_client.chat.completions.create.assert_not_called()


def test_empty_response(mock_client):
    mock_client.chat.completions.create.return_value.choices[
        0
    ].message.content = ""

    provider = OpenAIProvider(client=mock_client)

    with pytest.raises(ValueError, match="empty response"):
        provider.generate(
            GenerationRequest(
                model=MODEL,
                prompt="Hello",
            )
        )


def test_factory_registers_openai(mock_client):
    registry = build_provider_registry()

    assert registry.available() == [
        "groq",
        "ollama",
        "openai",
    ]

    provider = create_provider(
        "openai",
        openai_client=mock_client,
    )

    assert isinstance(provider, OpenAIProvider)