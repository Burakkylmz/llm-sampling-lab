
import math
from typing import Any


SUPPORTED_EXPERIMENTS = {
    "temperature",
    "top_p",
    "top_k",
}


def _summary_map(
    report: dict[str, Any],
) -> dict[float, dict[str, Any]]:
    """Index diversity statistics by parameter value."""

    summary = report.get("summary")

    if not isinstance(summary, list) or not summary:
        raise ValueError(
            "Report has no valid diversity summary."
        )

    indexed = {}

    for item in summary:
        if not isinstance(item, dict):
            raise ValueError("Invalid summary entry.")

        value = item.get("parameter_value")

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid parameter value.")

        ratio = item.get("unique_response_ratio")

        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or not 0 <= ratio <= 1
        ):
            raise ValueError("Invalid unique response ratio.")

        key = float(value)

        if key in indexed:
            raise ValueError("Duplicate parameter value.")

        indexed[key] = item

    return indexed


def compare_experiment_reports(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare two completed experiments of the same type."""

    experiment_type = report_a.get("experiment_type")

    if (
        experiment_type not in SUPPORTED_EXPERIMENTS
        or experiment_type != report_b.get("experiment_type")
    ):
        raise ValueError(
            "Experiments must use the same sampling strategy."
        )

    if (
        report_a.get("status") != "completed"
        or report_b.get("status") != "completed"
    ):
        raise ValueError(
            "Only completed experiments can be compared."
        )

    summary_a = _summary_map(report_a)
    summary_b = _summary_map(report_b)

    values = sorted(set(summary_a) | set(summary_b))

    comparison = []

    for value in values:
        item_a = summary_a.get(value, {})
        item_b = summary_b.get(value, {})

        comparison.append({
            "Parameter": (
                int(value)
                if experiment_type == "top_k"
                else value
            ),
            "Experiment A": item_a.get(
                "unique_response_ratio"
            ),
            "Experiment B": item_b.get(
                "unique_response_ratio"
            ),
            "Total A": item_a.get("total_responses"),
            "Total B": item_b.get("total_responses"),
            "Unique A": item_a.get("unique_responses"),
            "Unique B": item_b.get("unique_responses"),
        })

    return comparison


def compare_multiple_experiment_reports(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare two or more completed sampling experiments."""

    if len(reports) < 2:
        raise ValueError(
            "At least two experiments are required."
        )

    experiment_type = reports[0].get("experiment_type")

    if experiment_type not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            "Unsupported sampling experiment type."
        )

    for report in reports:
        if report.get("experiment_type") != experiment_type:
            raise ValueError(
                "All experiments must use the same "
                "sampling strategy."
            )

        if report.get("status") != "completed":
            raise ValueError(
                "Only completed experiments can be compared."
            )

    summaries = [
        _summary_map(report)
        for report in reports
    ]

    parameter_values = sorted(
        set().union(
            *(set(summary) for summary in summaries)
        )
    )

    comparison = []

    for value in parameter_values:
        row = {
            "Parameter": (
                int(value)
                if experiment_type == "top_k"
                else value
            )
        }

        for index, summary in enumerate(
            summaries,
            start=1,
        ):
            item = summary.get(value)

            row[f"Experiment {index}"] = (
                item["unique_response_ratio"]
                if item is not None
                else None
            )

        comparison.append(row)

    return comparison