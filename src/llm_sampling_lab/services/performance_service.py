
import math
from collections import defaultdict
from collections.abc import Mapping
from statistics import mean, median
from typing import Any


PARAMETER_FIELDS = {
    "temperature": "temperature",
    "top-p": "top_p",
    "top_p": "top_p",
    "top-k": "top_k",
    "top_k": "top_k",
}


def _get_field(item: Any, field: str) -> Any:
    """Read a field from an object or saved JSON record."""
    if isinstance(item, Mapping):
        return item.get(field)
    return getattr(item, field, None)


def _valid_latency(value: Any) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Invalid latency_seconds.")

    return float(value)


def _valid_tokens(value: Any) -> int | None:
    if value is None:
        return None

    if type(value) is not int or value < 0:
        raise ValueError("Invalid token count.")

    return value


def build_performance_rows(
    results: list[Any],
    strategy: str,
) -> list[dict[str, Any]]:
    """Normalize live results and historical JSON records."""

    if strategy not in PARAMETER_FIELDS:
        raise ValueError(
            f"Unsupported performance strategy: {strategy}"
        )

    parameter_field = PARAMETER_FIELDS[strategy]
    rows = []

    for index, result in enumerate(results, start=1):
        parameter = _get_field(result, parameter_field)

        if (
            isinstance(parameter, bool)
            or not isinstance(parameter, (int, float))
            or not math.isfinite(parameter)
        ):
            raise ValueError("Invalid sampling parameter.")

        rows.append({
            "Request": index,
            "Parameter": parameter,
            "Repetition": _get_field(
                result, "repetition"
            ),
            "Latency (s)": _valid_latency(
                _get_field(result, "latency_seconds")
            ),
            "Input Tokens": _valid_tokens(
                _get_field(result, "input_tokens")
            ),
            "Output Tokens": _valid_tokens(
                _get_field(result, "output_tokens")
            ),
        })

    return rows



def calculate_p95(
    latencies: list[float],
) -> float | None:
    """Calculate P95 latency using the nearest-rank method."""

    if not latencies:
        return None

    ordered = sorted(latencies)

    index = math.ceil(0.95 * len(ordered)) - 1

    return round(ordered[index], 4)


def summarize_performance_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate performance metrics by sampling parameter."""

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["Parameter"]].append(row)

    summary = []

    for parameter in sorted(grouped):
        group = grouped[parameter]

        latencies = [
            item["Latency (s)"]
            for item in group
            if item["Latency (s)"] is not None
        ]

        inputs = [
            item["Input Tokens"]
            for item in group
            if item["Input Tokens"] is not None
        ]

        outputs = [
            item["Output Tokens"]
            for item in group
            if item["Output Tokens"] is not None
        ]

        # Sum only requests with both usage components recorded.
        totals = [
            item["Input Tokens"] + item["Output Tokens"]
            for item in group
            if (
                item["Input Tokens"] is not None
                and item["Output Tokens"] is not None
            )
        ]

        summary.append({
            "Parameter": parameter,
            "Requests": len(group),
            "Measured Requests": len(latencies),
            "Average Latency (s)": (
                round(mean(latencies), 4)
                if latencies else None
            ),
            "Median Latency (s)": (
                round(median(latencies), 4)
                if latencies else None
            ),
            "P95 Latency (s)": calculate_p95(latencies),
            "Recorded Input Tokens": (
                sum(inputs) if inputs else None
            ),
            "Average Input Tokens / Request": (
                round(mean(inputs), 2) if inputs else None
            ),
            "Recorded Output Tokens": (
                sum(outputs) if outputs else None
            ),
            "Average Output Tokens / Request": (
                round(mean(outputs), 2) if outputs else None
            ),
            "Recorded Total Tokens": (
                sum(totals) if totals else None
            ),
            "Average Total Tokens / Request": (
                round(mean(totals), 2) if totals else None
            ),
            "Input Coverage": len(inputs),
            "Output Coverage": len(outputs),
            "Total Token Coverage": len(totals),
        })

    return summary