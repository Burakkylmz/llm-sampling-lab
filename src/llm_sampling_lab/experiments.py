
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from groq import Groq

from llm_sampling_lab.groq_client import generate_response


SamplingParameter = Literal["temperature", "top_p"]

ResultCallback = Callable[["SamplingResult"], None]


@dataclass(frozen=True)
class SamplingResult:
    """Store a single sampling experiment result."""

    experiment: str
    temperature: float
    top_p: float
    repetition: int
    response: str


def _run_experiment(
    client: Groq,
    model: str,
    prompt: str,
    parameter: SamplingParameter,
    values: list[float],
    fixed_value: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:
    """Execute sampling experiments with one varying parameter."""

    results: list[SamplingResult] = []

    for value in values:
        for repetition in range(1, repetitions + 1):

            temperature = (
                value if parameter == "temperature"
                else fixed_value
            )

            top_p = (
                value if parameter == "top_p"
                else fixed_value
            )

            response = generate_response(
                client=client,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
            )

            result = SamplingResult(
                experiment=parameter,
                temperature=temperature,
                top_p=top_p,
                repetition=repetition,
                response=response,
            )

            results.append(result)

            # Persist each result immediately when a callback exists.
            if on_result is not None:
                on_result(result)

    return results


def run_temperature_experiment(
    client: Groq,
    model: str,
    prompt: str,
    temperature_values: list[float],
    top_p: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:
    """Run temperature experiments with a fixed top-p."""

    return _run_experiment(
        client=client,
        model=model,
        prompt=prompt,
        parameter="temperature",
        values=temperature_values,
        fixed_value=top_p,
        repetitions=repetitions,
        on_result=on_result,
    )


def run_top_p_experiment(
    client: Groq,
    model: str,
    prompt: str,
    top_p_values: list[float],
    temperature: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:
    """Run top-p experiments with a fixed temperature."""

    return _run_experiment(
        client=client,
        model=model,
        prompt=prompt,
        parameter="top_p",
        values=top_p_values,
        fixed_value=temperature,
        repetitions=repetitions,
        on_result=on_result,
    )
