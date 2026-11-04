
import pytest

from llm_sampling_lab.domain.models import SamplingResult
from llm_sampling_lab.services.performance_service import (
    build_performance_rows,
    summarize_performance_rows,
)


def make_result(top_k, latency, input_tokens, output_tokens):
    return SamplingResult(
        experiment="top_k",
        temperature=0.8,
        top_p=0.9,
        top_k=top_k,
        repetition=1,
        response="Hello",
        latency_seconds=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_grouped_performance():
    results = [
        make_result(1, 0.1, 10, 5),
        make_result(1, 0.3, 10, 7),
        make_result(20, 0.2, 15, 8),
    ]

    rows = build_performance_rows(results, "top-k")
    summary = summarize_performance_rows(rows)

    assert len(summary) == 2
    assert summary[0]["Requests"] == 2
    assert summary[0]["Average Latency (s)"] == 0.2
    assert summary[0]["Recorded Input Tokens"] == 20
    assert summary[0]["Recorded Output Tokens"] == 12


def test_historical_json_support():
    records = [
        {
            "top_k": 40,
            "repetition": 1,
            "latency_seconds": 0.15,
            "input_tokens": 25,
            "output_tokens": 6,
        }
    ]

    rows = build_performance_rows(records, "top_k")

    assert rows[0]["Parameter"] == 40
    assert rows[0]["Latency (s)"] == 0.15


def test_old_reports_without_metrics():
    records = [
        {
            "top_k": 20,
            "repetition": 1,
        }
    ]

    rows = build_performance_rows(records, "top_k")
    summary = summarize_performance_rows(rows)

    assert summary[0]["Average Latency (s)"] is None
    assert summary[0]["Recorded Input Tokens"] is None


def test_partial_token_coverage():
    results = [
        make_result(20, 0.1, 10, 5),
        make_result(20, 0.2, None, None),
    ]

    rows = build_performance_rows(results, "top-k")
    summary = summarize_performance_rows(rows)

    assert summary[0]["Input Coverage"] == 1
    assert summary[0]["Output Coverage"] == 1


def test_reject_invalid_latency():
    with pytest.raises(ValueError, match="latency"):
        build_performance_rows(
            [make_result(1, -0.5, 10, 5)],
            "top-k",
        )


def test_reject_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        build_performance_rows([], "unknown")


def test_p95_latency():
    from llm_sampling_lab.services.performance_service import (
        calculate_p95,
    )

    latencies = [i / 100 for i in range(1, 21)]

    assert calculate_p95(latencies) == 0.19
    assert calculate_p95([]) is None
    assert calculate_p95([0.5]) == 0.5



def test_per_request_token_averages_and_combined_usage():
    results = [
        make_result(20, 0.1, 10, 5),
        make_result(20, 0.2, 30, 15),
    ]
    rows = build_performance_rows(results, "top-k")
    summary = summarize_performance_rows(rows)[0]

    assert summary["Recorded Input Tokens"] == 40
    assert summary["Average Input Tokens / Request"] == 20
    assert summary["Recorded Output Tokens"] == 20
    assert summary["Average Output Tokens / Request"] == 10
    assert summary["Recorded Total Tokens"] == 60
    assert summary["Average Total Tokens / Request"] == 30
    assert summary["Total Token Coverage"] == 2


def test_per_request_averages_use_only_observed_usage():
    results = [
        make_result(20, 0.1, 10, 5),
        make_result(20, 0.2, 20, None),
        make_result(20, 0.3, None, 15),
        make_result(20, 0.4, None, None),
    ]
    rows = build_performance_rows(results, "top-k")
    summary = summarize_performance_rows(rows)[0]

    assert summary["Requests"] == 4
    assert summary["Input Coverage"] == 2
    assert summary["Output Coverage"] == 2
    assert summary["Total Token Coverage"] == 1
    assert summary["Average Input Tokens / Request"] == 15
    assert summary["Average Output Tokens / Request"] == 10
    assert summary["Recorded Total Tokens"] == 15
    assert summary["Average Total Tokens / Request"] == 15


def test_per_request_averages_unknown_when_no_usage():
    rows = build_performance_rows(
        [make_result(20, 0.1, None, None)],
        "top-k",
    )
    summary = summarize_performance_rows(rows)[0]

    assert summary["Average Input Tokens / Request"] is None
    assert summary["Average Output Tokens / Request"] is None
    assert summary["Average Total Tokens / Request"] is None
    assert summary["Total Token Coverage"] == 0
