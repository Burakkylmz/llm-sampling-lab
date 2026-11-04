
from statistics import mean, median
from typing import Any

from llm_sampling_lab.services.performance_service import (
    build_performance_rows,
    calculate_p95,
    summarize_performance_rows,
)


SUPPORTED_STRATEGIES = {
    "temperature",
    "top_p",
    "top_k",
}


def build_benchmark_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Create a benchmark summary from a saved experiment."""

    if report.get("status") != "completed":
        raise ValueError(
            "Only completed experiments can be benchmarked."
        )

    strategy = report.get("experiment_type")

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "Unsupported experiment strategy."
        )

    results = report.get("results")

    if not isinstance(results, list):
        raise ValueError(
            "Experiment results must be a list."
        )

    parameters = report.get("parameters", {})

    if not isinstance(parameters, dict):
        raise ValueError(
            "Experiment parameters must be a dictionary."
        )

    rows = build_performance_rows(
        results,
        strategy,
    )

    parameter_summary = summarize_performance_rows(
        rows
    )

    latencies = [
        row["Latency (s)"]
        for row in rows
        if row["Latency (s)"] is not None
    ]

    input_tokens = [
        row["Input Tokens"]
        for row in rows
        if row["Input Tokens"] is not None
    ]

    output_tokens = [
        row["Output Tokens"]
        for row in rows
        if row["Output Tokens"] is not None
    ]

    complete_tokens = [
        row["Input Tokens"] + row["Output Tokens"]
        for row in rows
        if (
            row["Input Tokens"] is not None
            and row["Output Tokens"] is not None
        )
    ]

    total = len(rows)

    metrics = {
        "Requests": total,
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
            sum(input_tokens) if input_tokens else None
        ),
        "Average Input Tokens / Request": (
            round(mean(input_tokens), 2) if input_tokens else None
        ),
        "Recorded Output Tokens": (
            sum(output_tokens) if output_tokens else None
        ),
        "Average Output Tokens / Request": (
            round(mean(output_tokens), 2) if output_tokens else None
        ),
        "Recorded Total Tokens": (
            sum(complete_tokens) if complete_tokens else None
        ),
        "Average Total Tokens / Request": (
            round(mean(complete_tokens), 2) if complete_tokens else None
        ),
        "Latency Coverage": len(latencies),
        "Input Coverage": len(input_tokens),
        "Output Coverage": len(output_tokens),
        "Total Token Coverage": len(complete_tokens),
    }

    metadata = {
        "experiment_id": report.get("experiment_id"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "strategy": strategy,
        "prompt": report.get("prompt"),
        "started_at_utc": report.get("started_at_utc"),
        "values": parameters.get("values"),
        "fixed": parameters.get("fixed"),
        "repetitions": parameters.get("repetitions"),
        "warmup_requests": parameters.get(
            "warmup_requests"
        ),
        "seed": parameters.get("seed"),
        "randomize_order": parameters.get(
            "randomize_order"
        ),
        "order_seed": parameters.get("order_seed"),
    }

    return {
        "metadata": metadata,
        "metrics": metrics,
        "by_parameter": parameter_summary,
    }


def compare_benchmark_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare benchmark measurements from multiple reports."""

    if len(reports) < 2:
        raise ValueError(
            "At least two experiments are required."
        )

    # Validate strategies BEFORE processing performance data.
    strategies = {
        report.get("experiment_type")
        for report in reports
    }

    if len(strategies) != 1:
        raise ValueError(
            "Experiments must use the same sampling strategy."
        )

    benchmark_reports = [
        build_benchmark_report(report)
        for report in reports
    ]

    summary_rows = []
    parameter_rows = []

    for index, benchmark in enumerate(
        benchmark_reports,
        start=1,
    ):
        label = f"Experiment {index}"

        metadata = benchmark["metadata"]
        metrics = benchmark["metrics"]

        summary_rows.append({
            "Experiment": label,
            "Provider": metadata["provider"],
            "Model": metadata["model"],
            **metrics,
        })

        for group in benchmark["by_parameter"]:
            parameter_rows.append({
                "Experiment": label,
                "Provider": metadata["provider"],
                "Model": metadata["model"],
                **group,
            })

    warnings = []

    metadata_list = [
        item["metadata"]
        for item in benchmark_reports
    ]

    def differs(field: str) -> bool:
        values = [
            item.get(field)
            for item in metadata_list
        ]

        return any(
            value != values[0]
            for value in values[1:]
        )

    if differs("prompt"):
        warnings.append(
            "Prompts differ between experiments."
        )

    if differs("values"):
        warnings.append(
            "Tested sampling values differ."
        )

    if differs("fixed"):
        warnings.append(
            "Fixed sampling parameters differ."
        )

    if differs("repetitions"):
        warnings.append(
            "Repetition counts differ."
        )

    if differs("warmup_requests"):
        warnings.append(
            "Warm-up settings differ."
        )

    if differs("seed"):
        warnings.append(
            "Generation seed settings differ."
        )

    if differs("randomize_order"):
        warnings.append(
            "Execution ordering methods differ."
        )

    request_counts = [
        item["metrics"]["Requests"]
        for item in benchmark_reports
    ]

    if len(set(request_counts)) > 1:
        warnings.append(
            "Request counts differ. Compare average tokens per "
            "measured request rather than token totals; "
            "latency statistics may also be affected."
        )

    for index, benchmark in enumerate(
        benchmark_reports,
        start=1,
    ):
        metrics = benchmark["metrics"]

        if (
            metrics["Measured Requests"]
            < metrics["Requests"]
        ):
            warnings.append(
                f"Experiment {index}: latency coverage "
                "is incomplete."
            )

        if (
            metrics["Input Coverage"] < metrics["Requests"]
            or metrics["Output Coverage"] < metrics["Requests"]
        ):
            warnings.append(
                f"Experiment {index}: token coverage "
                "is incomplete."
            )

        if metrics["Measured Requests"] < 20:
            warnings.append(
                f"Experiment {index}: P95 is based on fewer "
                "than 20 measured requests and may be unstable."
            )

    return {
        "experiment_count": len(benchmark_reports),
        "summary_rows": summary_rows,
        "parameter_rows": parameter_rows,
        "warnings": warnings,
        "reports": benchmark_reports,
    }


def build_token_comparison_rows(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Show comparable per-request token usage against experiment 1.

    The percentage difference is relative to the first selected
    experiment's average combined token usage, not its token total.
    """

    if not summary_rows:
        return []

    reference = summary_rows[0].get(
        "Average Total Tokens / Request"
    )
    rows = []

    for index, item in enumerate(summary_rows, start=1):
        average = item.get("Average Total Tokens / Request")
        requests = item["Requests"]
        measured = item["Total Token Coverage"]

        if index == 1:
            change = "Reference"
        elif reference is None or reference == 0 or average is None:
            change = "N/A"
        else:
            difference = (average - reference) / reference * 100
            change = f"{difference:+.2f}%"

        provider = item.get("Provider") or "Unknown provider"
        model = item.get("Model") or "Unknown model"

        rows.append({
            "Provider / Model": f"{provider} / {model} (E{index})",
            "Requests": requests,
            "Recorded Total Tokens": item.get("Recorded Total Tokens"),
            "Average Total Tokens / Request": average,
            "Difference vs E1": change,
            "Token Coverage": f"{measured}/{requests}",
        })

    return rows
