"""Coverage for deterministic confidence estimation and guarded comparisons."""

from copy import deepcopy
from random import getstate

import pytest

from llm_sampling_lab.services.benchmark_quality_service import (
    bootstrap_mean_ci,
    iqr_outlier_summary,
    build_benchmark_quality_report,
    compare_benchmark_mean_latency,
)


def sample(top_k, latency):
    return {
        "top_k": top_k,
        "latency_seconds": latency,
        "input_tokens": 10,
        "output_tokens": 5,
        "repetition": 1,
    }


def report(latencies, *, experiment_id="a", model="llama3.2:1b"):
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "experiment_type": "top_k",
        "provider": "ollama",
        "model": model,
        "prompt": "Generate a slogan",
        "environment": {
            "python_version": "3.12.0",
            "operating_system": "Linux",
            "architecture": "x86_64",
        },
        "parameters": {
            "values": [20],
            "fixed": {"temperature": 0.8, "top_p": 0.9},
            "repetitions": len(latencies),
            "warmup_requests": 1,
            "seed": None,
            "randomize_order": True,
            "order_seed": 42,
            "max_output_tokens": 128,
        },
        "results": [sample(20, latency) for latency in latencies],
    }


def test_bootstrap_reproducible_and_local_rng():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    original = values.copy()
    rng_state = getstate()
    first = bootstrap_mean_ci(values, seed=42)
    second = bootstrap_mean_ci(values, seed=42)
    assert first == second
    assert values == original
    assert getstate() == rng_state
    assert first["CI Lower (s)"] <= first["Mean (s)"] <= first["CI Upper (s)"]


def test_constant_data_has_zero_width_interval():
    result = bootstrap_mean_ci([0.25] * 8)
    assert result["Mean (s)"] == 0.25
    assert result["CI Lower (s)"] == 0.25
    assert result["CI Upper (s)"] == 0.25


def test_small_sample_does_not_claim_confidence():
    result = bootstrap_mean_ci([0.1, 0.2, 0.3])
    assert result["Samples"] == 3
    assert result["CI Lower (s)"] is None
    assert result["CI Upper (s)"] is None
    assert bootstrap_mean_ci([])["Mean (s)"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 1},
        {"confidence_level": 0},
        {"confidence_level": float("nan")},
        {"resamples": 50},
        {"resamples": True},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_bootstrap_configuration(kwargs):
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1] * 5, **kwargs)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_invalid_latency_rejected(value):
    with pytest.raises(ValueError, match="nonnegative"):
        bootstrap_mean_ci([value] * 5)


def test_iqr_outlier_flagging_does_not_remove_samples():
    values = [1.0, 1.0, 1.0, 1.0, 20.0]
    original = values.copy()
    summary = iqr_outlier_summary(values)
    assert summary["Outlier Count"] == 1
    assert summary["Outlier Values (s)"] == [20.0]
    assert values == original
    assert iqr_outlier_summary([0.1, 0.2, 0.3])["Outlier Count"] is None


def test_quality_report_coverage_and_parameter_grouping():
    results = [
        sample(1, latency)
        for latency in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ] + [
        sample(20, latency)
        for latency in (0.2, 0.3, 0.4, 0.5, 0.6, None)
    ]
    quality = build_benchmark_quality_report(results, "top_k")
    assert quality["overall"]["Requests"] == 12
    assert quality["overall"]["Measured Requests"] == 11
    assert quality["by_parameter"][0]["Parameter"] == 1
    assert quality["by_parameter"][1]["Measured Requests"] == 5
    assert quality["by_parameter"][1]["CI Lower (s)"] is not None
    assert quality["by_parameter"][1]["Warnings"]
    assert quality == build_benchmark_quality_report(results, "top_k")


def test_quality_report_handles_missing_all_samples():
    quality = build_benchmark_quality_report(
        [sample(1, None)], "top_k"
    )
    assert quality["overall"]["Mean (s)"] is None
    assert quality["overall"]["CI Lower (s)"] is None
    assert quality["overall"]["Outlier Count"] is None


def test_comparable_mean_difference_is_exploratory():
    first = report([0.1] * 20, experiment_id="one")
    second = report([0.2] * 20, experiment_id="two")
    result = compare_benchmark_mean_latency(first, second)
    assert result["comparable"] is True
    assert result["overall"]["Delta Mean Latency (s)"] == 0.1
    assert result["overall"]["CI Lower (s)"] == 0.1
    assert result["overall"]["CI Upper (s)"] == 0.1
    assert len(result["by_parameter"]) == 1
    assert result["by_parameter"][0]["Estimate"] is not None
    assert result == compare_benchmark_mean_latency(first, second)


def test_missing_token_budget_refuses_comparison():
    first = report([0.1] * 10)
    second = deepcopy(first)
    second["parameters"].pop("max_output_tokens")
    result = compare_benchmark_mean_latency(first, second)
    assert result["comparable"] is False
    assert any("max_output_tokens" in issue for issue in result["reasons"])


def test_environment_and_model_mismatch_refused():
    first = report([0.1] * 10)
    second = report([0.2] * 10, model="another-model")
    second["environment"]["architecture"] = "arm64"
    result = compare_benchmark_mean_latency(first, second)
    assert result["comparable"] is False
    assert any("model" in reason for reason in result["reasons"])
    assert any("environments differ" in reason for reason in result["reasons"])


def test_incomplete_coverage_refuses_mean_difference():
    first = report([0.1] * 10)
    second = report([0.2] * 9 + [None])
    comparison = compare_benchmark_mean_latency(first, second)
    assert comparison["comparable"] is False
    assert any("Incomplete latency" in item for item in comparison["reasons"])


def test_small_sample_comparison_has_no_interval():
    first = report([0.1] * 3)
    second = report([0.2] * 3)
    comparison = compare_benchmark_mean_latency(first, second)
    assert comparison["comparable"] is True
    assert comparison["overall"] is None
    assert comparison["by_parameter"][0]["Estimate"] is None
    assert any("five" in w for w in comparison["warnings"])


def test_refuse_changed_generation_seed():
    first = report([0.1] * 10)
    second = deepcopy(first)
    second["parameters"]["seed"] = 42
    comparison = compare_benchmark_mean_latency(first, second)
    assert comparison["comparable"] is False
    assert any("seed" in msg for msg in comparison["reasons"])


def test_refuse_different_status():
    first = report([0.1] * 10)
    second = deepcopy(first)
    second["status"] = "failed"
    comparison = compare_benchmark_mean_latency(first, second)
    assert comparison["comparable"] is False
    assert any("completed" in msg for msg in comparison["reasons"])
