
import pytest

from llm_sampling_lab.services.comparison_service import (
    compare_multiple_experiment_reports,
)


def make_report(
    values,
    experiment_type="top_k",
    status="completed",
):
    return {
        "experiment_type": experiment_type,
        "status": status,
        "summary": [
            {
                "parameter_value": value,
                "total_responses": 10,
                "unique_responses": unique,
                "unique_response_ratio": unique / 10,
            }
            for value, unique in values
        ],
    }


def test_three_experiment_comparison():
    reports = [
        make_report([(1, 1), (20, 6)]),
        make_report([(1, 2), (20, 7)]),
        make_report([(1, 3), (20, 8)]),
    ]

    result = compare_multiple_experiment_reports(
        reports
    )

    assert len(result) == 2

    assert result[0]["Parameter"] == 1
    assert result[0]["Experiment 1"] == 0.1
    assert result[0]["Experiment 2"] == 0.2
    assert result[0]["Experiment 3"] == 0.3


def test_missing_parameter_is_none():
    reports = [
        make_report([(1, 1)]),
        make_report([(20, 6)]),
        make_report([(40, 8)]),
    ]

    result = compare_multiple_experiment_reports(
        reports
    )

    assert len(result) == 3

    assert result[0]["Experiment 2"] is None
    assert result[1]["Experiment 1"] is None
    assert result[2]["Experiment 3"] == 0.8


def test_requires_two_experiments():
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        compare_multiple_experiment_reports([
            make_report([(1, 1)])
        ])


def test_rejects_mixed_strategies():
    reports = [
        make_report([(1, 1)], "top_k"),
        make_report([(0.5, 5)], "temperature"),
    ]

    with pytest.raises(
        ValueError,
        match="same sampling strategy",
    ):
        compare_multiple_experiment_reports(
            reports
        )


def test_rejects_failed_experiments():
    reports = [
        make_report([(1, 1)]),
        make_report(
            [(1, 2)],
            status="failed",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="completed",
    ):
        compare_multiple_experiment_reports(
            reports
        )