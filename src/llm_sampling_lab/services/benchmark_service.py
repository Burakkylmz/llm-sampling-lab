
from statistics import mean, median
from typing import Any

from llm_sampling_lab.domain.models import SamplingResult


def summarize_benchmark(
    results: list[SamplingResult],
) -> dict[str, Any]:
    """Summarize latency and token usage for completed requests."""

    latencies = [
        result.latency_seconds
        for result in results
        if result.latency_seconds is not None
    ]

    input_tokens = [
        result.input_tokens
        for result in results
        if result.input_tokens is not None
    ]

    output_tokens = [
        result.output_tokens
        for result in results
        if result.output_tokens is not None
    ]

    combined_tokens = [
        result.input_tokens + result.output_tokens
        for result in results
        if (
            result.input_tokens is not None
            and result.output_tokens is not None
        )
    ]

    return {
        "total_requests": len(results),
        "measured_requests": len(latencies),
        "average_latency_seconds": (
            mean(latencies) if latencies else None
        ),
        "median_latency_seconds": (
            median(latencies) if latencies else None
        ),
        "total_input_tokens": (
            sum(input_tokens) if input_tokens else None
        ),
        "average_input_tokens_per_request": (
            round(mean(input_tokens), 2) if input_tokens else None
        ),
        "total_output_tokens": (
            sum(output_tokens) if output_tokens else None
        ),
        "average_output_tokens_per_request": (
            round(mean(output_tokens), 2) if output_tokens else None
        ),
        "total_tokens": (
            sum(combined_tokens) if combined_tokens else None
        ),
        "average_tokens_per_request": (
            round(mean(combined_tokens), 2) if combined_tokens else None
        ),
        "requests_with_input_usage": len(input_tokens),
        "requests_with_output_usage": len(output_tokens),
        "requests_with_complete_usage": len(combined_tokens),
    }
