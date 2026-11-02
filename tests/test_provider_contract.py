
import pytest

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.providers.registry import (
    ProviderRegistry,
)


class FakeProvider(BaseProvider):

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider="fake",
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.UNSUPPORTED,
        )

    def _generate(self, request):
        return GenerationResponse(
            provider="fake",
            model=request.model,
            text="Hello!",
        )


def test_provider_registry():
    registry = ProviderRegistry()
    registry.register("fake", FakeProvider)

    provider = registry.create("fake")

    assert isinstance(provider, FakeProvider)
    assert registry.available() == ["fake"]


def test_generation():
    provider = FakeProvider()

    result = provider.generate(
        GenerationRequest(
            model="test-model",
            prompt="Say hello",
            temperature=0.7,
        )
    )

    assert result.text == "Hello!"


def test_unsupported_top_k():
    provider = FakeProvider()

    with pytest.raises(ValueError, match="top_k"):
        provider.generate(
            GenerationRequest(
                model="test-model",
                prompt="Hello",
                top_k=5,
            )
        )


def test_invalid_top_p():
    provider = FakeProvider()

    with pytest.raises(ValueError, match="top_p"):
        provider.generate(
            GenerationRequest(
                model="test-model",
                prompt="Hello",
                top_p=1.5,
            )
        )


def test_unknown_provider():
    registry = ProviderRegistry()

    with pytest.raises(ValueError, match="Unknown provider"):
        registry.create("missing")
