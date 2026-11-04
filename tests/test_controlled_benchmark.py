
import json

import pytest

from llm_sampling_lab.domain.models import GenerationResponse
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.services.controlled_benchmark import (
    ControlledBenchmarkConfig,
    run_controlled_benchmark,
)


class FakeProvider(BaseProvider):
    name = "fake"
    supports_seed = True

    def __init__(self, fail_at_call=None):
        self.requests = []
        self.fail_at_call = fail_at_call

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider=self.name,
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.SUPPORTED,
        )

    def _generate(self, request):
        self.requests.append(request)

        if len(self.requests) == self.fail_at_call:
            raise RuntimeError("Simulated provider failure")

        return GenerationResponse(
            provider=self.name,
            model=request.model,
            text=f"Response {len(self.requests)}",
            input_tokens=10,
            output_tokens=5,
        )


def make_config(**overrides):
    settings = {
        "strategy": "top_k",
        "prompt": "Generate a slogan",
        "values": (1, 20, 40),
        "repetitions": 2,
        "warmup_requests": 1,
        "randomize_order": True,
        "order_seed": 42,
    }

    settings.update(overrides)
    return ControlledBenchmarkConfig(**settings)


def test_preview_does_not_call_provider():
    config = make_config()

    preview = config.preview()

    assert preview["recorded_requests"] == 6
    assert preview["warmup_requests"] == 1
    assert preview["total_provider_calls"] == 7
    assert len(preview["execution_order"]) == 6


def test_controlled_benchmark_execution(tmp_path):
    provider = FakeProvider()
    config = make_config()

    outcome = run_controlled_benchmark(
        provider=provider,
        model="test-model",
        config=config,
        output_dir=tmp_path,
    )

    assert len(provider.requests) == 7
    assert len(outcome.results) == 6
    assert outcome.report_path.exists()

    report = json.loads(
        outcome.report_path.read_text(encoding="utf-8")
    )

    assert report["status"] == "completed"
    assert len(report["results"]) == 6

    assert report["parameters"]["warmup_requests"] == 1
    assert report["parameters"]["randomize_order"] is True
    assert report["parameters"]["order_seed"] == 42

    expected = [
        (
            item["parameter_value"],
            item["repetition"],
        )
        for item in config.preview()["execution_order"]
    ]

    actual = [
        (item["top_k"], item["repetition"])
        for item in report["results"]
    ]

    assert actual == expected

    metrics = outcome.benchmark_report["metrics"]

    assert metrics["Requests"] == 6
    assert metrics["Measured Requests"] == 6
    assert metrics["Recorded Input Tokens"] == 60
    assert metrics["Recorded Output Tokens"] == 30


@pytest.mark.parametrize(
    "strategy,values",
    [
        ("temperature", (0.2, 0.8)),
        ("top_p", (0.3, 0.9)),
        ("top_k", (1, 20)),
    ],
)
def test_all_sampling_strategies(
    tmp_path,
    strategy,
    values,
):
    config = make_config(
        strategy=strategy,
        values=values,
        repetitions=1,
        warmup_requests=0,
        randomize_order=False,
        order_seed=None,
    )

    outcome = run_controlled_benchmark(
        provider=FakeProvider(),
        model="test-model",
        config=config,
        output_dir=tmp_path,
    )

    assert len(outcome.results) == 2

    assert (
        outcome.benchmark_report["metadata"]["strategy"]
        == strategy
    )


def test_order_seed_is_not_generation_seed(tmp_path):
    provider = FakeProvider()

    config = make_config(
        generation_seed=None,
        order_seed=42,
    )

    run_controlled_benchmark(
        provider=provider,
        model="test-model",
        config=config,
        output_dir=tmp_path,
    )

    assert all(
        request.seed is None
        for request in provider.requests
    )


def test_failed_run_preserves_completed_requests(tmp_path):
    provider = FakeProvider(fail_at_call=3)

    config = make_config(
        values=(1, 20),
        repetitions=1,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated provider failure",
    ):
        run_controlled_benchmark(
            provider=provider,
            model="test-model",
            config=config,
            output_dir=tmp_path,
        )

    files = list(tmp_path.glob("top_k/*.json"))

    assert len(files) == 1

    report = json.loads(
        files[0].read_text(encoding="utf-8")
    )

    assert report["status"] == "failed"
    assert len(report["results"]) == 1
    assert report["error_type"] == "RuntimeError"


def test_unsupported_generation_seed_rejected(tmp_path):
    provider = FakeProvider()
    provider.supports_seed = False

    config = make_config(generation_seed=42)

    with pytest.raises(ValueError, match="seed"):
        run_controlled_benchmark(
            provider=provider,
            model="test-model",
            config=config,
            output_dir=tmp_path,
        )

    assert list(tmp_path.glob("**/*.json")) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"values": ()},
        {"values": (1, 1)},
        {"repetitions": 0},
        {"warmup_requests": -1},
        {"order_seed": None},
        {"strategy": "top_p", "values": (0.0, 1.1)},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        make_config(**changes)
