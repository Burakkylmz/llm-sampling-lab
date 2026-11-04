
import json

import pytest

from llm_sampling_lab.domain.models import (
    GenerationResponse,
)
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.results import ExperimentRecorder
from llm_sampling_lab.services.execution_schedule import (
    build_execution_schedule,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


class FakeProvider(BaseProvider):
    supports_seed = True

    def __init__(self):
        self.requests = []

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider="fake",
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.SUPPORTED,
        )

    def _generate(self, request):
        self.requests.append(request)

        return GenerationResponse(
            provider="fake",
            model=request.model,
            text=f"Response {len(self.requests)}",
            input_tokens=10,
            output_tokens=5,
        )


def test_randomized_top_k_execution():
    provider = FakeProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        randomize_order=True,
        order_seed=42,
    )

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20, 40],
        temperature=0.8,
        top_p=0.9,
        repetitions=3,
    )

    expected = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=3,
        randomize=True,
        order_seed=42,
    )

    actual = [
        (result.top_k, result.repetition)
        for result in results
    ]

    assert actual == expected
    assert len(results) == 9

    assert [
        request.top_k for request in provider.requests
    ] == [
        value for value, _ in expected
    ]


def test_randomized_temperature_execution():
    provider = FakeProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        randomize_order=True,
        order_seed=123,
    )

    results = service.run_temperature(
        prompt="Hello",
        temperatures=[0.2, 0.8],
        top_p=0.9,
        repetitions=3,
    )

    expected = build_execution_schedule(
        values=[0.2, 0.8],
        repetitions=3,
        randomize=True,
        order_seed=123,
    )

    assert [
        (result.temperature, result.repetition)
        for result in results
    ] == expected


def test_warmup_not_in_randomized_results():
    provider = FakeProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        warmup_requests=1,
        randomize_order=True,
        order_seed=42,
    )

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20, 40],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
    )

    assert len(provider.requests) == 7
    assert len(results) == 6


def test_order_seed_does_not_set_generation_seed():
    provider = FakeProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        randomize_order=True,
        order_seed=42,
    )

    service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20],
        temperature=0.8,
        top_p=0.9,
        repetitions=1,
    )

    assert all(
        request.seed is None
        for request in provider.requests
    )


def test_randomization_metadata_saved(tmp_path):
    provider = FakeProvider()

    recorder = ExperimentRecorder(
        model="test-model",
        provider="fake",
        prompt="Hello",
        experiment_type="top_k",
        parameter_values=[1, 20],
        fixed_parameter="temperature",
        fixed_value=0.8,
        extra_fixed_parameters={"top_p": 0.9},
        repetitions=2,
        randomize_order=True,
        order_seed=42,
        output_dir=tmp_path,
    )

    service = ExperimentService(
        provider=provider,
        model="test-model",
        randomize_order=True,
        order_seed=42,
    )

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
        on_result=recorder.record,
    )

    recorder.complete()

    report = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert report["parameters"]["randomize_order"] is True
    assert report["parameters"]["order_seed"] == 42
    assert len(report["results"]) == 4

    assert [
        (item["top_k"], item["repetition"])
        for item in report["results"]
    ] == [
        (item.top_k, item.repetition)
        for item in results
    ]


def test_randomization_requires_order_seed():
    with pytest.raises(ValueError, match="order_seed"):
        ExperimentService(
            provider=FakeProvider(),
            model="test-model",
            randomize_order=True,
        )
