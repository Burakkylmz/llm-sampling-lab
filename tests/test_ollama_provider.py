
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import GenerationRequest
from llm_sampling_lab.providers.capabilities import SupportStatus
from llm_sampling_lab.providers.factory import (
    build_provider_registry,
    create_provider,
)
from llm_sampling_lab.providers.ollama_provider import OllamaProvider


MODEL = "llama3.2:1b"


@pytest.fixture
def mock_client():
    client = MagicMock()

    client.list.return_value = SimpleNamespace(
        models=[
            SimpleNamespace(model=MODEL),
        ]
    )

    client.show.return_value = SimpleNamespace(
        capabilities=["completion"],
    )

    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content="Hello from Ollama"
        ),
        done_reason="stop",
        prompt_eval_count=12,
        eval_count=8,
    )

    return client


def test_model_discovery(mock_client):
    provider = OllamaProvider(client=mock_client)

    models = provider.discover_models()

    assert len(models) == 1
    assert models[0].model_id == MODEL
    assert models[0].provider == "ollama"


def test_native_top_k_capability(mock_client):
    provider = OllamaProvider(client=mock_client)

    caps = provider.get_capabilities(MODEL)

    assert caps.temperature == SupportStatus.SUPPORTED
    assert caps.top_p == SupportStatus.SUPPORTED
    assert caps.top_k == SupportStatus.SUPPORTED


def test_native_top_k_generation(mock_client):
    provider = OllamaProvider(client=mock_client)

    result = provider.generate(
        GenerationRequest(
            model=MODEL,
            prompt="Write a short slogan",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=128,
        )
    )

    assert result.text == "Hello from Ollama"
    assert result.input_tokens == 12
    assert result.output_tokens == 8

    kwargs = mock_client.chat.call_args.kwargs

    assert kwargs["model"] == MODEL
    assert kwargs["stream"] is False
    assert kwargs["options"]["temperature"] == 0.7
    assert kwargs["options"]["top_p"] == 0.9
    assert kwargs["options"]["top_k"] == 40
    assert kwargs["options"]["num_predict"] == 128


def test_invalid_top_k_rejected(mock_client):
    provider = OllamaProvider(client=mock_client)

    with pytest.raises(ValueError, match="top_k"):
        provider.generate(
            GenerationRequest(
                model=MODEL,
                prompt="Hello",
                top_k=-1,
            )
        )

    mock_client.chat.assert_not_called()


def test_unknown_model(mock_client):
    provider = OllamaProvider(client=mock_client)

    caps = provider.get_capabilities("missing-model")

    assert caps.top_k == SupportStatus.UNKNOWN


def test_non_generation_model(mock_client):
    mock_client.show.return_value = SimpleNamespace(
        capabilities=["embedding"],
    )

    provider = OllamaProvider(client=mock_client)

    caps = provider.get_capabilities(MODEL)

    assert caps.top_k == SupportStatus.UNSUPPORTED


def test_factory_registers_ollama(mock_client):
    registry = build_provider_registry()

    assert registry.available() == [
        "groq", "ollama", "openai"
    ]

    provider = create_provider(
        "ollama",
        ollama_client=mock_client,
    )

    assert isinstance(provider, OllamaProvider)