
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
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


class FakeProvider(BaseProvider):

    def __init__(self):
        self.requests = []

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider="fake",
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
        )

    def _generate(self, request: GenerationRequest):
        self.requests.append(request)

        return GenerationResponse(
            provider="fake",
            model=request.model,
            text=f"Response {len(self.requests)}",
        )


def create_service():
    provider = FakeProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    return service, provider


def test_temperature_experiment():
    service, provider = create_service()

    results = service.run_temperature(
        prompt="Generate a slogan",
        temperatures=[0.2, 0.8],
        top_p=1.0,
        repetitions=2,
    )

    assert len(results) == 4
    assert len(provider.requests) == 4

    assert [r.temperature for r in results] == [
        0.2, 0.2, 0.8, 0.8
    ]

    assert all(r.top_p == 1.0 for r in results)
    assert [r.repetition for r in results] == [
        1, 2, 1, 2
    ]


def test_top_p_experiment():
    service, provider = create_service()

    results = service.run_top_p(
        prompt="Generate a slogan",
        top_p_values=[0.3, 0.9],
        temperature=0.7,
        repetitions=2,
    )

    assert len(results) == 4

    assert [r.top_p for r in results] == [
        0.3, 0.3, 0.9, 0.9
    ]

    assert all(
        r.temperature == 0.7 for r in results
    )


def test_result_callback():
    service, _ = create_service()
    recorded = []

    service.run_temperature(
        prompt="Hello",
        temperatures=[0.5],
        top_p=1.0,
        repetitions=3,
        on_result=recorded.append,
    )

    assert len(recorded) == 3


def test_invalid_repetitions():
    service, provider = create_service()

    with pytest.raises(ValueError, match="repetitions"):
        service.run_temperature(
            prompt="Hello",
            temperatures=[0.5],
            top_p=1.0,
            repetitions=0,
        )

    assert provider.requests == []


def test_empty_parameter_values():
    service, provider = create_service()

    with pytest.raises(ValueError, match="empty"):
        service.run_temperature(
            prompt="Hello",
            temperatures=[],
            top_p=1.0,
            repetitions=2,
        )

    assert provider.requests == []
