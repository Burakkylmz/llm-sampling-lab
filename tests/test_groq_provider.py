
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import GenerationRequest
from llm_sampling_lab.providers.capabilities import SupportStatus
from llm_sampling_lab.providers.groq_provider import GroqProvider
from llm_sampling_lab.providers.registry import ProviderRegistry


MODEL = "openai/gpt-oss-20b"


@pytest.fixture
def mock_client():
    client = MagicMock()
    completion = client.chat.completions.create.return_value

    completion.choices[0].message.content = "Test response"
    completion.choices[0].finish_reason = "stop"
    completion.usage.prompt_tokens = 12
    completion.usage.completion_tokens = 8

    return client


def test_capabilities():
    provider = GroqProvider()

    capabilities = provider.get_capabilities(MODEL)

    assert capabilities.temperature == SupportStatus.SUPPORTED
    assert capabilities.top_p == SupportStatus.SUPPORTED
    assert capabilities.top_k == SupportStatus.UNSUPPORTED


def test_generation(mock_client):
    provider = GroqProvider(client=mock_client)

    response = provider.generate(
        GenerationRequest(
            model=MODEL,
            prompt="Generate a slogan",
            temperature=0.7,
            top_p=0.9,
        )
    )

    assert response.provider == "groq"
    assert response.text == "Test response"
    assert response.finish_reason == "stop"
    assert response.input_tokens == 12
    assert response.output_tokens == 8

    kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 256


def test_unsupported_top_k(mock_client):
    provider = GroqProvider(client=mock_client)

    with pytest.raises(ValueError, match="top_k"):
        provider.generate(
            GenerationRequest(
                model=MODEL,
                prompt="Hello",
                top_k=5,
            )
        )

    mock_client.chat.completions.create.assert_not_called()


def test_unknown_model_capabilities():
    provider = GroqProvider()

    capabilities = provider.get_capabilities("unverified-model")

    assert capabilities.temperature == SupportStatus.UNKNOWN


def test_unknown_model_parameter_rejected(mock_client):
    provider = GroqProvider(client=mock_client)

    with pytest.raises(ValueError, match="unknown"):
        provider.generate(
            GenerationRequest(
                model="unverified-model",
                prompt="Hello",
                temperature=0.7,
            )
        )

    mock_client.chat.completions.create.assert_not_called()


def test_empty_response(mock_client):
    mock_client.chat.completions.create.return_value.choices[
        0
    ].message.content = ""

    provider = GroqProvider(client=mock_client)

    with pytest.raises(ValueError, match="empty response"):
        provider.generate(
            GenerationRequest(
                model=MODEL,
                prompt="Hello",
            )
        )


def test_registry_integration():
    registry = ProviderRegistry()
    registry.register("groq", GroqProvider)

    provider = registry.create("groq")

    assert isinstance(provider, GroqProvider)
