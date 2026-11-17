"""Regression tests for descriptive benchmark statistics and metadata."""

import json
import math

import pytest

from llm_sampling_lab.domain.models import GenerationResponse
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.results import ExperimentRecorder
from llm_sampling_lab.services.benchmark_reporting_service import (
    build_benchmark_report,
)
from llm_sampling_lab.services.controlled_benchmark import (
    ControlledBenchmarkConfig,
    run_controlled_benchmark,
)
from llm_sampling_lab.services.environment_metadata import (
    capture_environment_metadata,
)
from llm_sampling_lab.services.statistical_reliability_service import (
    build_statistical_reliability_report,
    nearest_rank_percentile,
)


def record(top_k: int, latency: float | None) -> dict:
    return {
        "top_k": top_k,
        "latency_seconds": latency,
        "input_tokens": 10,
        "output_tokens": 5,
        "response": "test response",
        "repetition": 1,
    }


def test_nearest_rank_percentiles():
    values = [i / 100 for i in range(1, 21)]
    assert nearest_rank_percentile(values, 50) == 0.1
    assert nearest_rank_percentile(values, 90) == 0.18
    assert nearest_rank_percentile(values, 95) == 0.19
    assert nearest_rank_percentile(values, 99) == 0.2


def test_small_sample_tail_percentiles_equal_maximum():
    values = [0.3, 0.1, 0.2]
    assert nearest_rank_percentile(values, 50) == 0.2
    assert nearest_rank_percentile(values, 95) == 0.3
    assert nearest_rank_percentile(values, 99) == 0.3


def test_empty_sample_returns_none():
    assert nearest_rank_percentile([], 50) is None
    assert nearest_rank_percentile([], 99) is None


@pytest.mark.parametrize("percentile", [0, -1, 101, True, float("nan")])
def test_reject_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentile"):
        nearest_rank_percentile([0.1], percentile)


@pytest.mark.parametrize("latency", [-1, float("nan"), math.inf, True])
def test_reject_invalid_latency_sample(latency):
    with pytest.raises(ValueError, match="latency"):
        nearest_rank_percentile([latency], 95)


def test_grouped_latency_and_dispersion():
    results = [
        record(1, 0.1),
        record(1, 0.2),
        record(20, 0.3),
        record(20, 0.4),
    ]

    report = build_statistical_reliability_report(
        results, "top_k"
    )
    assert report["percentile_method"] == "nearest_rank"
    assert report["overall"]["Requests"] == 4
    assert report["overall"]["Measured Requests"] == 4
    assert report["overall"]["Latency Coverage (%)"] == 100.0
    assert report["overall"]["Average Latency (s)"] == 0.25
    assert report["overall"]["Std Dev Latency (s)"] == 0.1291
    assert len(report["by_parameter"]) == 2
    assert report["by_parameter"][0]["Parameter"] == 1
    assert report["by_parameter"][0]["P50 Latency (s)"] == 0.1
    assert report["by_parameter"][1]["P99 Latency (s)"] == 0.4
    assert report["by_parameter"][0]["CV (%)"] is not None


def test_partial_and_missing_latency_coverage():
    report = build_statistical_reliability_report(
        [record(20, 0.1), record(20, None)], "top_k"
    )
    overall = report["overall"]
    assert overall["Measured Requests"] == 1
    assert overall["Latency Coverage (%)"] == 50.0
    assert overall["Std Dev Latency (s)"] is None
    assert any("missing" in w.lower() for w in overall["Warnings"])

    empty = build_statistical_reliability_report(
        [record(20, None)], "top_k"
    )["overall"]
    assert empty["Measured Requests"] == 0
    assert empty["P95 Latency (s)"] is None
    assert empty["P99 Latency (s)"] is None
    assert empty["CV (%)"] is None


def test_empty_results_are_supported():
    report = build_statistical_reliability_report([], "top_k")
    assert report["overall"]["Requests"] == 0
    assert report["overall"]["Latency Coverage (%)"] is None
    assert report["by_parameter"] == []


def test_zero_latency_has_no_cv():
    overall = build_statistical_reliability_report(
        [record(1, 0.0), record(1, 0.0)], "top_k"
    )["overall"]
    assert overall["CV (%)"] is None
    assert overall["P95 Latency (s)"] == 0.0


def test_invalid_latency_and_strategy_rejected():
    with pytest.raises(ValueError, match="latency"):
        build_statistical_reliability_report(
            [record(1, -0.1)], "top_k"
        )
    with pytest.raises(ValueError, match="strategy"):
        build_statistical_reliability_report([], "invalid")


def test_environment_capture_does_not_include_identifiers():
    metadata = capture_environment_metadata()
    assert set(metadata) == {
        "python_version",
        "operating_system",
        "os_release",
        "architecture",
        "execution_context",
    }
    assert all(isinstance(v, str) for v in metadata.values())
    assert metadata["python_version"]
    assert metadata["execution_context"] in {"container", "unknown"}


def test_recorder_environment_is_optional_and_persisted(tmp_path):
    common = dict(
        model="test-model",
        prompt="Hello",
        experiment_type="top_k",
        parameter_values=[1],
        fixed_parameter="temperature",
        fixed_value=0.8,
        repetitions=1,
        output_dir=tmp_path,
    )
    with_env = ExperimentRecorder(
        **common,
        environment={"python_version": "3.12.0"},
    )
    data = json.loads(with_env.file_path.read_text(encoding="utf-8"))
    assert data["environment"] == {"python_version": "3.12.0"}

    without_env = ExperimentRecorder(**common)
    assert "environment" not in without_env.report


def test_legacy_report_without_environment_still_works():
    old = {
        "status": "completed",
        "experiment_type": "top_k",
        "provider": "ollama",
        "model": "llama3.2:1b",
        "parameters": {},
        "results": [record(20, 0.1), record(20, 0.2)],
    }
    benchmark = build_benchmark_report(old)
    assert benchmark["metadata"]["environment"] is None
    assert benchmark["statistical_reliability"]["overall"]["Requests"] == 2
    assert benchmark["statistical_reliability"]["overall"]["P95 Latency (s)"] == 0.2


class FakeProvider(BaseProvider):
    name = "fake"

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider=self.name,
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.SUPPORTED,
        )

    def _generate(self, request):
        return GenerationResponse(
            provider=self.name,
            model=request.model,
            text="Hello",
            input_tokens=10,
            output_tokens=5,
        )


def test_controlled_benchmark_saves_environment_and_statistics(tmp_path):
    outcome = run_controlled_benchmark(
        provider=FakeProvider(),
        model="test-model",
        config=ControlledBenchmarkConfig(
            strategy="top_k",
            prompt="Hello",
            values=(1, 20),
            repetitions=2,
            warmup_requests=0,
            randomize_order=False,
            order_seed=None,
        ),
        output_dir=tmp_path,
    )
    raw = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert raw["status"] == "completed"
    assert raw["environment"]["python_version"]
    assert raw["environment"]["execution_context"] in {"container", "unknown"}

    benchmark = outcome.benchmark_report
    assert benchmark["metadata"]["environment"] == raw["environment"]
    assert benchmark["statistical_reliability"]["overall"]["Requests"] == 4
    assert len(benchmark["statistical_reliability"]["by_parameter"]) == 2
