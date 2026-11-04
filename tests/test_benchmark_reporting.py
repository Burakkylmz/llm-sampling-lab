
import pytest

from llm_sampling_lab.services.benchmark_reporting_service import (
    build_benchmark_report,
    compare_benchmark_reports,
)


def make_report(
    *,
    experiment_id="experiment-a",
    prompt="Generate a slogan",
    latencies=(0.1, 0.2, 0.3),
    input_tokens=10,
    output_tokens=5,
):
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "experiment_type": "top_k",
        "provider": "ollama",
        "model": "llama3.2:1b",
        "prompt": prompt,
        "started_at_utc": "2026-11-04T12:00:00+00:00",
        "parameters": {
            "values": [20],
            "fixed": {
                "temperature": 0.8,
                "top_p": 0.9,
            },
            "repetitions": len(latencies),
            "warmup_requests": 1,
            "seed": None,
            "randomize_order": True,
            "order_seed": 42,
        },
        "results": [
            {
                "experiment": "top_k",
                "top_k": 20,
                "repetition": index,
                "response": f"Response {index}",
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            for index, latency in enumerate(
                latencies,
                start=1,
            )
        ],
    }


def test_single_benchmark_summary():
    report = build_benchmark_report(
        make_report()
    )

    metrics = report["metrics"]

    assert metrics["Requests"] == 3
    assert metrics["Measured Requests"] == 3
    assert metrics["Average Latency (s)"] == 0.2
    assert metrics["Median Latency (s)"] == 0.2
    assert metrics["P95 Latency (s)"] == 0.3
    assert metrics["Recorded Input Tokens"] == 30
    assert metrics["Recorded Output Tokens"] == 15


def test_benchmark_metadata():
    report = build_benchmark_report(
        make_report()
    )

    metadata = report["metadata"]

    assert metadata["provider"] == "ollama"
    assert metadata["seed"] is None
    assert metadata["randomize_order"] is True
    assert metadata["order_seed"] == 42
    assert metadata["warmup_requests"] == 1


def test_two_experiment_comparison():
    first = make_report()

    second = make_report(
        experiment_id="experiment-b",
        latencies=(0.2, 0.3, 0.4),
    )

    comparison = compare_benchmark_reports([
        first,
        second,
    ])

    assert comparison["experiment_count"] == 2
    assert len(comparison["summary_rows"]) == 2
    assert len(comparison["parameter_rows"]) == 2

    assert (
        comparison["summary_rows"][0]["Average Latency (s)"]
        == 0.2
    )

    assert (
        comparison["summary_rows"][1]["Average Latency (s)"]
        == 0.3
    )


def test_different_prompts_generate_warning():
    first = make_report()

    second = make_report(
        experiment_id="experiment-b",
        prompt="Write a product description",
    )

    comparison = compare_benchmark_reports([
        first,
        second,
    ])

    assert any(
        "Prompts differ" in warning
        for warning in comparison["warnings"]
    )


def test_missing_benchmark_data():
    report = make_report()

    for result in report["results"]:
        result.pop("latency_seconds")
        result.pop("input_tokens")
        result.pop("output_tokens")

    benchmark = build_benchmark_report(report)

    metrics = benchmark["metrics"]

    assert metrics["Average Latency (s)"] is None
    assert metrics["P95 Latency (s)"] is None
    assert metrics["Recorded Input Tokens"] is None
    assert metrics["Input Coverage"] == 0


def test_reject_incomplete_experiment():
    report = make_report()
    report["status"] = "failed"

    with pytest.raises(
        ValueError,
        match="completed",
    ):
        build_benchmark_report(report)


def test_reject_mixed_strategies():
    first = make_report()
    second = make_report(
        experiment_id="experiment-b"
    )

    second["experiment_type"] = "temperature"

    with pytest.raises(
        ValueError,
        match="same sampling strategy",
    ):
        compare_benchmark_reports([
            first,
            second,
        ])


def test_requires_two_experiments():
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        compare_benchmark_reports([
            make_report()
        ])



def test_benchmark_token_averages():
    report = build_benchmark_report(
        make_report(latencies=(0.1, 0.2, 0.3))
    )
    metrics = report["metrics"]
    assert metrics["Recorded Total Tokens"] == 45
    assert metrics["Average Input Tokens / Request"] == 10
    assert metrics["Average Output Tokens / Request"] == 5
    assert metrics["Average Total Tokens / Request"] == 15
    assert metrics["Total Token Coverage"] == 3

    group = report["by_parameter"][0]
    assert group["Recorded Total Tokens"] == 45
    assert group["Average Total Tokens / Request"] == 15
    assert group["Total Token Coverage"] == 3


def test_benchmark_partial_usage_uses_measured_denominator():
    report = make_report()
    report["results"][1]["output_tokens"] = None
    report["results"][2]["input_tokens"] = None

    metrics = build_benchmark_report(report)["metrics"]

    assert metrics["Requests"] == 3
    assert metrics["Input Coverage"] == 2
    assert metrics["Output Coverage"] == 2
    assert metrics["Total Token Coverage"] == 1
    assert metrics["Average Input Tokens / Request"] == 10
    assert metrics["Average Output Tokens / Request"] == 5
    assert metrics["Recorded Total Tokens"] == 15
    assert metrics["Average Total Tokens / Request"] == 15


def test_benchmark_without_usage_has_unknown_averages():
    report = make_report()
    for result in report["results"]:
        result["input_tokens"] = None
        result["output_tokens"] = None

    metrics = build_benchmark_report(report)["metrics"]

    assert metrics["Recorded Total Tokens"] is None
    assert metrics["Average Input Tokens / Request"] is None
    assert metrics["Average Output Tokens / Request"] is None
    assert metrics["Average Total Tokens / Request"] is None
    assert metrics["Total Token Coverage"] == 0



def test_different_request_counts_use_per_request_mean():
    long_report = make_report(
        experiment_id="long",
        latencies=(0.1,) * 20,
        input_tokens=60,
        output_tokens=40,
    )
    short_report = make_report(
        experiment_id="short",
        latencies=(0.1,) * 10,
        input_tokens=90,
        output_tokens=60,
    )

    comparison = compare_benchmark_reports([long_report, short_report])
    long_metrics, short_metrics = comparison["summary_rows"]

    assert long_metrics["Recorded Total Tokens"] == 2000
    assert short_metrics["Recorded Total Tokens"] == 1500
    assert long_metrics["Average Total Tokens / Request"] == 100
    assert short_metrics["Average Total Tokens / Request"] == 150
    assert any("Request counts differ" in w for w in comparison["warnings"])
