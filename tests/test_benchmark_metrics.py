
import json
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import GenerationResponse
from llm_sampling_lab.results import ExperimentRecorder
from llm_sampling_lab.services import experiment_service
from llm_sampling_lab.services.benchmark_service import (
    summarize_benchmark,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        (
            "temperature",
            {"temperatures": [0.8], "top_p": 0.9},
        ),
        (
            "top_p",
            {"top_p_values": [0.9], "temperature": 0.8},
        ),
        (
            "top_k",
            {
                "top_k_values": [20],
                "temperature": 0.8,
                "top_p": 0.9,
            },
        ),
    ],
)
def test_metrics_captured_for_all_strategies(
    monkeypatch,
    strategy,
    kwargs,
):
    provider = MagicMock()

    provider.generate.return_value = GenerationResponse(
        provider="test",
        model="test-model",
        text="Generated slogan",
        input_tokens=12,
        output_tokens=8,
    )

    times = iter([10.0, 10.25])

    monkeypatch.setattr(
        experiment_service,
        "perf_counter",
        lambda: next(times),
    )

    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    runner = {
        "temperature": service.run_temperature,
        "top_p": service.run_top_p,
        "top_k": service.run_top_k,
    }[strategy]

    results = runner(
        prompt="Generate a slogan",
        repetitions=1,
        **kwargs,
    )

    assert len(results) == 1

    result = results[0]

    assert result.latency_seconds == pytest.approx(0.25)
    assert result.input_tokens == 12
    assert result.output_tokens == 8


def test_benchmark_summary():
    provider = MagicMock()

    provider.generate.side_effect = [
        GenerationResponse(
            provider="test",
            model="test-model",
            text="First",
            input_tokens=10,
            output_tokens=5,
        ),
        GenerationResponse(
            provider="test",
            model="test-model",
            text="Second",
            input_tokens=20,
            output_tokens=7,
        ),
    ]

    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[20],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
    )

    summary = summarize_benchmark(results)

    assert summary["total_requests"] == 2
    assert summary["measured_requests"] == 2
    assert summary["total_input_tokens"] == 30
    assert summary["total_output_tokens"] == 12
    assert summary["average_input_tokens_per_request"] == 15
    assert summary["average_output_tokens_per_request"] == 6
    assert summary["total_tokens"] == 42
    assert summary["average_tokens_per_request"] == 21
    assert summary["requests_with_complete_usage"] == 2
    assert summary["average_latency_seconds"] >= 0


def test_missing_token_usage_remains_unknown():
    provider = MagicMock()

    provider.generate.return_value = GenerationResponse(
        provider="test",
        model="test-model",
        text="Hello",
    )

    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    results = service.run_temperature(
        prompt="Hello",
        temperatures=[0.5],
        top_p=1.0,
        repetitions=1,
    )

    summary = summarize_benchmark(results)

    assert summary["total_input_tokens"] is None
    assert summary["total_output_tokens"] is None
    assert summary["average_input_tokens_per_request"] is None
    assert summary["average_output_tokens_per_request"] is None
    assert summary["average_tokens_per_request"] is None
    assert summary["requests_with_complete_usage"] == 0


def test_metrics_are_persisted_to_json(tmp_path):
    provider = MagicMock()

    provider.generate.return_value = GenerationResponse(
        provider="ollama",
        model="test-model",
        text="Charge Ahead",
        input_tokens=15,
        output_tokens=6,
    )

    recorder = ExperimentRecorder(
        model="test-model",
        provider="ollama",
        prompt="Generate a slogan",
        experiment_type="top_k",
        parameter_values=[20],
        fixed_parameter="temperature",
        fixed_value=0.8,
        extra_fixed_parameters={"top_p": 0.9},
        repetitions=1,
        output_dir=tmp_path,
    )

    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    service.run_top_k(
        prompt="Generate a slogan",
        top_k_values=[20],
        temperature=0.8,
        top_p=0.9,
        repetitions=1,
        on_result=recorder.record,
    )

    recorder.complete()

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    result = data["results"][0]

    assert data["status"] == "completed"
    assert result["latency_seconds"] >= 0
    assert result["input_tokens"] == 15
    assert result["output_tokens"] == 6
