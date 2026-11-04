
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import GenerationRequest
from llm_sampling_lab.providers.capabilities import (
    SupportStatus,
)
from llm_sampling_lab.providers.groq_provider import (
    GroqProvider,
)
from llm_sampling_lab.providers.openai_provider import (
    OpenAIProvider,
)
from llm_sampling_lab.providers.model_catalog import (
    get_model_capabilities,
)


@pytest.mark.parametrize(
    "provider, model",
    [
        ("groq", "openai/gpt-oss-20b"),
        ("openai", "gpt-4.1-mini"),
    ],
)
def test_known_models(provider, model):
    capabilities = get_model_capabilities(
        provider, model
    )

    assert capabilities.temperature == SupportStatus.SUPPORTED
    assert capabilities.top_p == SupportStatus.SUPPORTED
    assert capabilities.top_k == SupportStatus.UNSUPPORTED


def test_unknown_model():
    capabilities = get_model_capabilities(
        "openai", "unverified-model"
    )

    assert capabilities.temperature == SupportStatus.UNKNOWN
    assert capabilities.top_k == SupportStatus.UNKNOWN


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_temperature_out_of_range(value):
    capabilities = get_model_capabilities(
        "groq", "openai/gpt-oss-20b"
    )

    with pytest.raises(ValueError, match="temperature"):
        capabilities.validate(temperature=value)


@pytest.mark.parametrize("value", [0.0, 2.0])
def test_temperature_boundaries(value):
    capabilities = get_model_capabilities(
        "openai", "gpt-4.1-mini"
    )

    capabilities.validate(temperature=value)


@pytest.mark.parametrize("value", [0.0, 1.1])
def test_top_p_out_of_range(value):
    capabilities = get_model_capabilities(
        "openai", "gpt-4.1-mini"
    )

    with pytest.raises(ValueError, match="top_p"):
        capabilities.validate(top_p=value)


def test_top_k_not_supported():
    capabilities = get_model_capabilities(
        "groq", "openai/gpt-oss-20b"
    )

    with pytest.raises(ValueError, match="top_k"):
        capabilities.validate(top_k=5)


@pytest.mark.parametrize(
    "provider_class, model",
    [
        (GroqProvider, "openai/gpt-oss-20b"),
        (OpenAIProvider, "gpt-4.1-mini"),
    ],
)
def test_invalid_parameter_never_calls_api(
    provider_class,
    model,
):
    client = MagicMock()
    provider = provider_class(client=client)

    with pytest.raises(ValueError, match="temperature"):
        provider.generate(
            GenerationRequest(
                model=model,
                prompt="Hello",
                temperature=3.0,
            )
        )

    client.chat.completions.create.assert_not_called()