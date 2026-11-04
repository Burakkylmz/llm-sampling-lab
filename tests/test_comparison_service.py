
import pytest

from llm_sampling_lab.services.comparison_service import (
    compare_experiment_reports,
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


def test_comparison_aligns_parameter_values():
    report_a = make_report([
        (1, 1),
        (20, 6),
    ])

    report_b = make_report([
        (20, 7),
        (40, 8),
    ])

    comparison = compare_experiment_reports(
        report_a,
        report_b,
    )

    assert [row["Parameter"] for row in comparison] == [
        1, 20, 40,
    ]

    assert comparison[0]["Experiment A"] == 0.1
    assert comparison[0]["Experiment B"] is None

    assert comparison[1]["Experiment A"] == 0.6
    assert comparison[1]["Experiment B"] == 0.7

    assert comparison[2]["Experiment A"] is None
    assert comparison[2]["Experiment B"] == 0.8


def test_reject_different_experiment_types():
    report_a = make_report(
        [(1, 1)],
        experiment_type="top_k",
    )

    report_b = make_report(
        [(0.5, 5)],
        experiment_type="temperature",
    )

    with pytest.raises(ValueError, match="same"):
        compare_experiment_reports(
            report_a,
            report_b,
        )


def test_reject_incomplete_experiment():
    report_a = make_report([(1, 1)])
    report_b = make_report(
        [(1, 2)],
        status="failed",
    )

    with pytest.raises(ValueError, match="completed"):
        compare_experiment_reports(
            report_a,
            report_b,
        )


def test_reject_empty_summary():
    report_a = make_report([])
    report_b = make_report([(1, 2)])

    with pytest.raises(ValueError, match="summary"):
        compare_experiment_reports(
            report_a,
            report_b,
        )


def test_reject_duplicate_parameter():
    report_a = make_report([
        (20, 3),
        (20, 5),
    ])
    report_b = make_report([(20, 4)])

    with pytest.raises(ValueError, match="Duplicate"):
        compare_experiment_reports(
            report_a,
            report_b,
        )