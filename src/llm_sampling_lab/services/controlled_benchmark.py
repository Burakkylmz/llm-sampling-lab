
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_sampling_lab.domain.models import SamplingResult
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.results import ExperimentRecorder
from llm_sampling_lab.services.benchmark_reporting_service import (
    build_benchmark_report,
)
from llm_sampling_lab.services.execution_schedule import (
    build_execution_schedule,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


Strategy = Literal["temperature", "top_p", "top_k"]


@dataclass(frozen=True)
class ControlledBenchmarkConfig:
    """Immutable configuration for a controlled benchmark."""

    strategy: Strategy
    prompt: str
    values: tuple[float | int, ...]

    repetitions: int = 30
    fixed_temperature: float = 0.8
    fixed_top_p: float = 0.9
    max_output_tokens: int = 128

    warmup_requests: int = 1
    randomize_order: bool = True
    order_seed: int | None = 42
    generation_seed: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {
            "temperature", "top_p", "top_k"
        }:
            raise ValueError("Unsupported benchmark strategy.")

        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError(
                "values must be a non-empty tuple."
            )

        for value in self.values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Invalid parameter value.")

            if self.strategy == "temperature" and value < 0:
                raise ValueError("Invalid temperature value.")

            if self.strategy == "top_p" and not 0 < value <= 1:
                raise ValueError("Invalid top_p value.")

            if self.strategy == "top_k" and (
                type(value) is not int or value < 1
            ):
                raise ValueError(
                    "Top-k values must be positive integers."
                )

        if len(set(self.values)) != len(self.values):
            raise ValueError(
                "Parameter values must be unique."
            )

        if (
            type(self.repetitions) is not int
            or self.repetitions < 1
        ):
            raise ValueError(
                "repetitions must be a positive integer."
            )

        if (
            type(self.warmup_requests) is not int
            or not 0 <= self.warmup_requests <= 3
        ):
            raise ValueError(
                "warmup_requests must be between 0 and 3."
            )

        if (
            isinstance(self.fixed_temperature, bool)
            or not isinstance(
                self.fixed_temperature, (int, float)
            )
            or not math.isfinite(self.fixed_temperature)
            or self.fixed_temperature < 0
        ):
            raise ValueError("Invalid fixed_temperature.")

        if (
            isinstance(self.fixed_top_p, bool)
            or not isinstance(self.fixed_top_p, (int, float))
            or not math.isfinite(self.fixed_top_p)
            or not 0 < self.fixed_top_p <= 1
        ):
            raise ValueError("Invalid fixed_top_p.")

        if (
            type(self.max_output_tokens) is not int
            or self.max_output_tokens < 1
        ):
            raise ValueError("Invalid max_output_tokens.")

        if self.generation_seed is not None and (
            type(self.generation_seed) is not int
            or self.generation_seed < 0
        ):
            raise ValueError("Invalid generation_seed.")

        # Reuse the existing schedule validator.
        build_execution_schedule(
            values=list(self.values),
            repetitions=self.repetitions,
            randomize=self.randomize_order,
            order_seed=self.order_seed,
        )

    def preview(self) -> dict[str, Any]:
        """Show planned requests without calling any provider."""

        schedule = build_execution_schedule(
            values=list(self.values),
            repetitions=self.repetitions,
            randomize=self.randomize_order,
            order_seed=self.order_seed,
        )

        return {
            "strategy": self.strategy,
            "recorded_requests": len(schedule),
            "warmup_requests": self.warmup_requests,
            "total_provider_calls": (
                len(schedule) + self.warmup_requests
            ),
            "execution_order": [
                {
                    "execution_index": index,
                    "parameter_value": value,
                    "repetition": repetition,
                }
                for index, (value, repetition) in enumerate(
                    schedule,
                    start=1,
                )
            ],
        }


@dataclass(frozen=True)
class ControlledBenchmarkOutcome:
    """Completed benchmark results and persisted report."""

    report_path: Path
    results: list[SamplingResult]
    benchmark_report: dict[str, Any]


def run_controlled_benchmark(
    *,
    provider: BaseProvider,
    model: str,
    config: ControlledBenchmarkConfig,
    output_dir: str | Path = "results",
    on_result: Callable[[SamplingResult], None] | None = None,
) -> ControlledBenchmarkOutcome:
    """Execute a controlled sampling benchmark."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model cannot be empty.")

    # Validate provider/seed compatibility before saving a report.
    service = ExperimentService(
        provider=provider,
        model=model,
        max_output_tokens=config.max_output_tokens,
        warmup_requests=config.warmup_requests,
        seed=config.generation_seed,
        randomize_order=config.randomize_order,
        order_seed=config.order_seed,
    )

    if config.strategy == "temperature":
        fixed_parameter = "top_p"
        fixed_value = config.fixed_top_p
        extra_fixed = None

    elif config.strategy == "top_p":
        fixed_parameter = "temperature"
        fixed_value = config.fixed_temperature
        extra_fixed = None

    else:
        fixed_parameter = "temperature"
        fixed_value = config.fixed_temperature
        extra_fixed = {
            "top_p": config.fixed_top_p
        }

    recorder = ExperimentRecorder(
        model=model,
        provider=provider.name,
        prompt=config.prompt,
        experiment_type=config.strategy,
        parameter_values=list(config.values),
        fixed_parameter=fixed_parameter,
        fixed_value=fixed_value,
        extra_fixed_parameters=extra_fixed,
        repetitions=config.repetitions,
        warmup_requests=config.warmup_requests,
        seed=config.generation_seed,
        randomize_order=config.randomize_order,
        order_seed=config.order_seed,
        output_dir=output_dir,
    )

    def record(result: SamplingResult) -> None:
        recorder.record(result)

        if on_result is not None:
            on_result(result)

    try:
        if config.strategy == "temperature":
            results = service.run_temperature(
                prompt=config.prompt,
                temperatures=list(config.values),
                top_p=config.fixed_top_p,
                repetitions=config.repetitions,
                on_result=record,
            )

        elif config.strategy == "top_p":
            results = service.run_top_p(
                prompt=config.prompt,
                top_p_values=list(config.values),
                temperature=config.fixed_temperature,
                repetitions=config.repetitions,
                on_result=record,
            )

        else:
            results = service.run_top_k(
                prompt=config.prompt,
                top_k_values=list(config.values),
                temperature=config.fixed_temperature,
                top_p=config.fixed_top_p,
                repetitions=config.repetitions,
                on_result=record,
            )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    benchmark_report = build_benchmark_report(
        recorder.report
    )

    return ControlledBenchmarkOutcome(
        report_path=recorder.file_path,
        results=results,
        benchmark_report=benchmark_report,
    )
