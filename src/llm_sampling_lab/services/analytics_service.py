
from collections import defaultdict
from typing import Any

from llm_sampling_lab.domain.models import SamplingResult
from llm_sampling_lab.services.comparison_service import (
    compare_multiple_experiment_reports,
)


STRATEGY_FIELDS = {
    "temperature": "temperature",
    "top-p": "top_p",
    "top-k": "top_k",
}


def summarize_sampling_results(
    results: list[SamplingResult],
    strategy: str,
) -> list[dict[str, Any]]:
    """Calculate group-level diversity and response counts."""

    if strategy not in STRATEGY_FIELDS:
        raise ValueError(f"Unknown strategy: {strategy}")

    parameter_name = STRATEGY_FIELDS[strategy]

    groups = defaultdict(list)

    for result in results:
        value = getattr(result, parameter_name)

        if value is None:
            raise ValueError(
                f"Missing sampling parameter: {parameter_name}"
            )

        groups[value].append(result.response)

    rows = []

    for value in sorted(groups):
        responses = groups[value]
        total = len(responses)
        unique = len(set(responses))
        ratio = unique / total

        rows.append({
            "Parameter": value,
            "Total Responses": total,
            "Unique Responses": unique,
            "Duplicate Responses": total - unique,
            "Unique Ratio": ratio,
            "Diversity (%)": round(ratio * 100, 1),
        })

    if rows:
        baseline = rows[0]["Diversity (%)"]

        for row in rows:
            row["Delta from baseline (pp)"] = round(
                row["Diversity (%)"] - baseline,
                1,
            )

    return rows


def build_comparison_insights(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare multi-experiment counts and diversity deltas."""

    comparison = compare_multiple_experiment_reports(reports)

    indexed_summaries = []

    for report in reports:
        indexed_summaries.append({
            float(item["parameter_value"]): item
            for item in report["summary"]
        })

    rows = []

    for item in comparison:
        parameter = item["Parameter"]
        baseline = item.get("Experiment 1")

        row = {"Parameter": parameter}

        for index, summary in enumerate(
            indexed_summaries,
            start=1,
        ):
            group = summary.get(float(parameter))
            ratio = item.get(f"Experiment {index}")

            row[f"E{index} Total"] = (
                group.get("total_responses")
                if group is not None
                else None
            )

            row[f"E{index} Unique"] = (
                group.get("unique_responses")
                if group is not None
                else None
            )

            row[f"E{index} Diversity (%)"] = (
                round(ratio * 100, 1)
                if ratio is not None
                else None
            )

            if index > 1:
                row[f"E{index} - E1 (pp)"] = (
                    round((ratio - baseline) * 100, 1)
                    if ratio is not None and baseline is not None
                    else None
                )

        rows.append(row)

    return rows