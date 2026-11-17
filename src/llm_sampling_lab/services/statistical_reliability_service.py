"""Descriptive latency statistics for controlled sampling experiments.

Percentiles use the nearest-rank definition, matching existing P95 metrics.
These descriptive statistics do not establish statistical significance.
"""

import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from llm_sampling_lab.services.performance_service import (
    build_performance_rows,
)


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    """Return a nearest-rank percentile, or None if no samples exist."""

    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not math.isfinite(percentile)
        or not 0 < percentile <= 100
    ):
        raise ValueError("Invalid percentile.")

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("Invalid latency value.")

    if not values:
        return None

    ordered = sorted(values)
    index = math.ceil(percentile / 100 * len(ordered)) - 1
    return round(ordered[index], 4)


def _summarize_group(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize validated request rows, retaining missingness."""

    latencies = [
        row["Latency (s)"]
        for row in rows
        if row["Latency (s)"] is not None
    ]

    count = len(latencies)
    total = len(rows)
    average = mean(latencies) if latencies else None
    deviation = stdev(latencies) if count >= 2 else None

    variation = (
        deviation / average * 100
        if deviation is not None and average is not None and average > 0
        else None
    )

    warnings = []

    if count < total:
        warnings.append(
            "Some requests are missing latency measurements."
        )

    if count == 0:
        warnings.append("No latency measurements are available.")
    else:
        if count < 20:
            warnings.append(
                "P95 is the sample maximum for fewer than "
                "20 observations; it may be unstable."
            )
        if count < 100:
            warnings.append(
                "P99 is the sample maximum for fewer than "
                "100 observations; it may be unstable."
            )

    return {
        "Requests": total,
        "Measured Requests": count,
        "Latency Coverage (%)": (
            round(100 * count / total, 1) if total else None
        ),
        "Average Latency (s)": (
            round(average, 4) if average is not None else None
        ),
        "Std Dev Latency (s)": (
            round(deviation, 4) if deviation is not None else None
        ),
        "CV (%)": round(variation, 2) if variation is not None else None,
        "P50 Latency (s)": nearest_rank_percentile(latencies, 50),
        "P90 Latency (s)": nearest_rank_percentile(latencies, 90),
        "P95 Latency (s)": nearest_rank_percentile(latencies, 95),
        "P99 Latency (s)": nearest_rank_percentile(latencies, 99),
        "Warnings": warnings,
    }


def build_statistical_reliability_report(
    results: list[Any],
    strategy: str,
) -> dict[str, Any]:
    """Analyze total and per-parameter request latency distributions.

    Accepts both SamplingResult objects and saved JSON record dictionaries.
    Uses the existing performance normalizer for identical validation rules.
    """

    rows = build_performance_rows(results, strategy)
    grouped: dict[float | int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row["Parameter"]].append(row)

    return {
        "strategy": strategy,
        "percentile_method": "nearest_rank",
        "overall": _summarize_group(rows),
        "by_parameter": [
            {"Parameter": parameter, **_summarize_group(grouped[parameter])}
            for parameter in sorted(grouped)
        ],
    }
