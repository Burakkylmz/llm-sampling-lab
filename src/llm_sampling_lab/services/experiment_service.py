
from collections.abc import Callable
from typing import Literal

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    SamplingResult,
)
from llm_sampling_lab.providers.base import BaseProvider


ExperimentType = Literal["temperature", "top_p"]
ResultCallback = Callable[[SamplingResult], None]


class ExperimentService:
    """Provider-independent sampling experiment engine."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        max_output_tokens: int = 256,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens

    def _run(
        self,
        *,
        experiment: ExperimentType,
        prompt: str,
        values: list[float],
        fixed_value: float,
        repetitions: int,
        on_result: ResultCallback | None = None,
    ) -> list[SamplingResult]:

        if not values:
            raise ValueError(
                "Experiment values cannot be empty."
            )

        if (
            type(repetitions) is not int
            or repetitions < 1
        ):
            raise ValueError(
                "repetitions must be a positive integer."
            )

        results: list[SamplingResult] = []

        for value in values:
            for repetition in range(1, repetitions + 1):

                temperature = (
                    value
                    if experiment == "temperature"
                    else fixed_value
                )

                top_p = (
                    value
                    if experiment == "top_p"
                    else fixed_value
                )

                request = GenerationRequest(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=self.max_output_tokens,
                )

                response = self.provider.generate(request)

                result = SamplingResult(
                    experiment=experiment,
                    temperature=temperature,
                    top_p=top_p,
                    repetition=repetition,
                    response=response.text,
                )

                results.append(result)

                if on_result is not None:
                    on_result(result)

        return results

    def run_temperature(
        self,
        *,
        prompt: str,
        temperatures: list[float],
        top_p: float,
        repetitions: int,
        on_result: ResultCallback | None = None,
    ) -> list[SamplingResult]:

        return self._run(
            experiment="temperature",
            prompt=prompt,
            values=temperatures,
            fixed_value=top_p,
            repetitions=repetitions,
            on_result=on_result,
        )

    def run_top_p(
        self,
        *,
        prompt: str,
        top_p_values: list[float],
        temperature: float,
        repetitions: int,
        on_result: ResultCallback | None = None,
    ) -> list[SamplingResult]:

        return self._run(
            experiment="top_p",
            prompt=prompt,
            values=top_p_values,
            fixed_value=temperature,
            repetitions=repetitions,
            on_result=on_result,
        )
