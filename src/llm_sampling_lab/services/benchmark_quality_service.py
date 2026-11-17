"""Descriptive benchmark quality metrics and cautious mean comparisons.

Bootstrap estimates assume independent request observations. A single benchmark
session can violate that assumption due to caching, time trends or load.
No p-values or definitive statistical-significance claims are produced.
"""

import math
from collections import defaultdict
from random import Random
from statistics import mean
from typing import Any

from llm_sampling_lab.services.performance_service import (
    build_performance_rows,
)

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42
MIN_CI_SAMPLES = 5


def _validated_latencies(values: list[float]) -> list[float]:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("Latency samples must be finite nonnegative numbers.")
    return [float(value) for value in values]


def _quantile(ordered: list[float], probability: float) -> float:
    """Linear interpolation quantile for already sorted, nonempty samples."""
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _validate_options(confidence_level: float, resamples: int, seed: int) -> None:
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be between zero and one.")
    if type(resamples) is not int or not 100 <= resamples <= 20000:
        raise ValueError("resamples must be an integer from 100 to 20000.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")


def _resampled_means(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    """IID nonparametric bootstrap with a local, repeatable random generator."""
    rng = Random(seed)
    sample_size = len(values)
    return [
        mean(rng.choices(values, k=sample_size))
        for _ in range(resamples)
    ]


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean (at least 5 observations).

    Returns no CI for undersized samples rather than a misleading interval.
    """

    _validate_options(confidence_level, resamples, seed)
    samples = _validated_latencies(values)
    result = {
        "Samples": len(samples),
        "Mean (s)": round(mean(samples), 4) if samples else None,
        "CI Lower (s)": None,
        "CI Upper (s)": None,
    }
    if len(samples) < MIN_CI_SAMPLES:
        return result

    distribution = sorted(
        _resampled_means(samples, resamples=resamples, seed=seed)
    )
    alpha = (1 - confidence_level) / 2
    result["CI Lower (s)"] = round(_quantile(distribution, alpha), 4)
    result["CI Upper (s)"] = round(_quantile(distribution, 1 - alpha), 4)
    return result


def iqr_outlier_summary(values: list[float]) -> dict[str, Any]:
    """Tukey 1.5x IQR fences; does not remove or modify observations."""
    samples = sorted(_validated_latencies(values))
    output = {
        "Outlier Count": None,
        "Lower Fence (s)": None,
        "Upper Fence (s)": None,
        "Outlier Values (s)": [],
    }
    if len(samples) < 4:
        return output
    q1 = _quantile(samples, 0.25)
    q3 = _quantile(samples, 0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = [
        round(value, 4)
        for value in samples
        if value < lower or value > upper
    ]
    output.update({
        "Outlier Count": len(flagged),
        "Lower Fence (s)": round(lower, 4),
        "Upper Fence (s)": round(upper, 4),
        "Outlier Values (s)": flagged,
    })
    return output


def _latencies(rows: list[dict[str, Any]]) -> list[float]:
    return [
        row["Latency (s)"]
        for row in rows
        if row["Latency (s)"] is not None
    ]


def _summarize(
    rows: list[dict[str, Any]],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    samples = _latencies(rows)
    ci = bootstrap_mean_ci(
        samples,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    outliers = iqr_outlier_summary(samples)
    warnings = []
    if len(samples) < len(rows):
        warnings.append("Latency coverage is incomplete.")
    if len(samples) < MIN_CI_SAMPLES:
        warnings.append(
            "At least five measured requests are needed for a bootstrap CI."
        )
    elif len(samples) < 20:
        warnings.append(
            "Fewer than 20 samples: bootstrap intervals may be unstable."
        )
    if len(samples) < 4:
        warnings.append(
            "At least four samples are needed for an IQR outlier check."
        )
    return {
        "Requests": len(rows),
        "Measured Requests": len(samples),
        **ci,
        **outliers,
        "Warnings": warnings,
    }


def build_benchmark_quality_report(
    results: list[Any],
    strategy: str,
    *,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Quality statistics overall and per sampling-parameter value."""
    _validate_options(confidence_level, resamples, seed)
    rows = build_performance_rows(results, strategy)
    groups: dict[float | int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["Parameter"]].append(row)

    return {
        "method": "iid_percentile_bootstrap",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "outlier_method": "Tukey_1.5_IQR",
        "overall": _summarize(
            rows,
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        ),
        "by_parameter": [
            {
                "Parameter": parameter,
                **_summarize(
                    groups[parameter],
                    confidence_level=confidence_level,
                    resamples=resamples,
                    seed=seed + index + 1,
                ),
            }
            for index, parameter in enumerate(sorted(groups))
        ],
        "interpretation": (
            "Bootstrap assumes independent observations. Outliers are flagged "
            "but are not removed. These estimates do not prove significance "
            "or control for provider load, caching or time trends."
        ),
    }


def _compatibility_issues(
    first: dict[str, Any],
    second: dict[str, Any],
) -> list[str]:
    """Require matching documented settings before estimating differences."""
    issues = []
    if first.get("status") != "completed" or second.get("status") != "completed":
        issues.append("Both experiments must be completed.")

    for key in ("experiment_type", "provider", "model", "prompt"):
        a, b = first.get(key), second.get(key)
        if a is None or b is None:
            issues.append(f"Missing {key} metadata.")
        elif a != b:
            issues.append(f"Different {key} values.")

    for key in (
        "values",
        "fixed",
        "repetitions",
        "warmup_requests",
        "seed",
        "randomize_order",
        "order_seed",
        "max_output_tokens",
    ):
        a = first.get("parameters") or {}
        b = second.get("parameters") or {}
        if key not in a or key not in b:
            issues.append(f"Missing {key} metadata.")
        elif a[key] != b[key]:
            issues.append(f"Different {key} settings.")

    a_env, b_env = first.get("environment"), second.get("environment")
    if not isinstance(a_env, dict) or not isinstance(b_env, dict):
        issues.append("Environment metadata is missing.")
    elif a_env != b_env:
        issues.append("Execution environments differ.")

    return issues


def _mean_difference_ci(
    first: list[float],
    second: list[float],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if min(len(first), len(second)) < MIN_CI_SAMPLES:
        return None
    first_means = _resampled_means(first, resamples=resamples, seed=seed)
    second_means = _resampled_means(
        second, resamples=resamples, seed=seed + 1
    )
    differences = sorted(
        b - a for a, b in zip(first_means, second_means)
    )
    alpha = (1 - confidence_level) / 2
    return {
        "Delta Mean Latency (s)": round(mean(second) - mean(first), 4),
        "CI Lower (s)": round(_quantile(differences, alpha), 4),
        "CI Upper (s)": round(_quantile(differences, 1 - alpha), 4),
        "Samples E1": len(first),
        "Samples E2": len(second),
    }


def compare_benchmark_mean_latency(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Exploratory, independent-bootstrap comparison (E2 minus E1).

    Incompatible or undocumented conditions return reasons, not estimates.
    No p-values or conclusions about significance are produced.
    """
    _validate_options(confidence_level, resamples, seed)
    reasons = _compatibility_issues(first, second)
    if reasons:
        return {
            "comparable": False,
            "reasons": reasons,
            "overall": None,
            "by_parameter": [],
        }

    strategy = first["experiment_type"]
    a_rows = build_performance_rows(first.get("results", []), strategy)
    b_rows = build_performance_rows(second.get("results", []), strategy)

    # Incomplete data can systematically bias a comparison.
    if len(_latencies(a_rows)) != len(a_rows) or len(_latencies(b_rows)) != len(b_rows):
        return {
            "comparable": False,
            "reasons": ["Incomplete latency coverage."],
            "overall": None,
            "by_parameter": [],
        }

    groups_a: dict[float | int, list[float]] = defaultdict(list)
    groups_b: dict[float | int, list[float]] = defaultdict(list)
    for row in a_rows:
        groups_a[row["Parameter"]].append(row["Latency (s)"])
    for row in b_rows:
        groups_b[row["Parameter"]].append(row["Latency (s)"])

    if set(groups_a) != set(groups_b):
        return {
            "comparable": False,
            "reasons": ["Measured sampling parameter groups differ."],
            "overall": None,
            "by_parameter": [],
        }

    warnings = []
    if min(len(a_rows), len(b_rows)) < 20:
        warnings.append(
            "Fewer than 20 requests per experiment; uncertainty may be high."
        )
    if min(len(a_rows), len(b_rows)) < MIN_CI_SAMPLES:
        warnings.append("At least five requests per experiment are required.")
    if any(
        min(len(groups_a[key]), len(groups_b[key])) < MIN_CI_SAMPLES
        for key in groups_a
    ):
        warnings.append(
            "At least five measurements per parameter group are required."
        )

    parameter_rows = []
    for index, key in enumerate(sorted(groups_a)):
        parameter_rows.append({
            "Parameter": key,
            "Estimate": _mean_difference_ci(
                groups_a[key], groups_b[key],
                confidence_level=confidence_level,
                resamples=resamples,
                seed=seed + index * 2 + 10,
            ),
        })

    return {
        "comparable": True,
        "reasons": [],
        "overall": _mean_difference_ci(
            _latencies(a_rows),
            _latencies(b_rows),
            confidence_level=confidence_level,
            resamples=resamples,
            seed=seed,
        ),
        "by_parameter": parameter_rows,
        "warnings": warnings,
        "method": "independent_iid_percentile_bootstrap",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "interpretation": (
            "E2 - E1; a negative value indicates lower observed mean latency "
            "in E2. Matching metadata does not guarantee equal runtime load "
            "or independent requests. This is exploratory, not causal."
        ),
    }
