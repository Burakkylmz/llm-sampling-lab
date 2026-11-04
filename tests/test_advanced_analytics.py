
import pytest

from llm_sampling_lab.domain.models import SamplingResult
from llm_sampling_lab.services.analytics_service import (
    summarize_sampling_results,
    build_comparison_insights,
)


def make_result(top_k, response):
    return SamplingResult(
        experiment="top_k",
        temperature=0.8,
        top_p=0.9,
        top_k=top_k,
        repetition=1,
        response=response,
    )


def make_report(unique_1, unique_20):
    return {
        "experiment_type": "top_k",
        "status": "completed",
        "summary": [
            {
                "parameter_value": 1,
                "total_responses": 10,
                "unique_responses": unique_1,
                "unique_response_ratio": unique_1 / 10,
            },
            {
                "parameter_value": 20,
                "total_responses": 10,
                "unique_responses": unique_20,
                "unique_response_ratio": unique_20 / 10,
            },
        ],
    }


def test_group_summary():
    results = [
        make_result(1, "A"),
        make_result(1, "A"),
        make_result(20, "B"),
        make_result(20, "C"),
    ]

    summary = summarize_sampling_results(
        results, "top-k"
    )

    assert len(summary) == 2
    assert summary[0]["Total Responses"] == 2
    assert summary[0]["Unique Responses"] == 1
    assert summary[0]["Diversity (%)"] == 50.0

    assert summary[1]["Unique Responses"] == 2
    assert summary[1]["Diversity (%)"] == 100.0
    assert summary[1]["Delta from baseline (pp)"] == 50.0


def test_empty_results():
    assert summarize_sampling_results([], "top-k") == []


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        summarize_sampling_results(
            [], "unsupported"
        )


def test_three_report_comparison():
    reports = [
        make_report(1, 6),
        make_report(2, 7),
        make_report(3, 8),
    ]

    result = build_comparison_insights(reports)

    assert len(result) == 2
    assert result[0]["E1 Total"] == 10
    assert result[0]["E2 - E1 (pp)"] == 10.0
    assert result[1]["E3 - E1 (pp)"] == 20.0


def test_reject_mixed_strategies():
    a = make_report(1, 6)
    b = make_report(2, 7)
    b["experiment_type"] = "temperature"

    with pytest.raises(ValueError):
        build_comparison_insights([a, b])
