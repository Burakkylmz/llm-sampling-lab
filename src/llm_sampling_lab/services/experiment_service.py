
from collections.abc import Callable
from time import perf_counter
from typing import Literal

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
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
        warmup_requests: int = 0,
        seed: int | None = None,
    ) -> None:
        if (
            type(warmup_requests) is not int
            or not 0 <= warmup_requests <= 3
        ):
            raise ValueError(
                "warmup_requests must be between 0 and 3."
            )

        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise ValueError(
                    "seed must be a non-negative integer."
                )

            if not getattr(provider, "supports_seed", False):
                raise ValueError(
                    "Selected provider does not support seed."
                )

        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.warmup_requests = warmup_requests
        self.seed = seed

    def _warm_up(
        self,
        request: GenerationRequest,
    ) -> None:
        """Warm up the selected model without recording results."""

        for _ in range(self.warmup_requests):
            self.provider.generate(request)

    def _generate_measured(
        self,
        request: GenerationRequest,
    ) -> tuple[GenerationResponse, float]:
        """Measure end-to-end provider generation latency."""

        start = perf_counter()
        response = self.provider.generate(request)
        latency = perf_counter() - start

        return response, latency

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

        # Use the first sampling configuration for warm-up.
        first_value = values[0]

        warmup_temperature = (
            first_value
            if experiment == "temperature"
            else fixed_value
        )

        warmup_top_p = (
            first_value
            if experiment == "top_p"
            else fixed_value
        )

        warmup_request = GenerationRequest(
            model=self.model,
            prompt=prompt,
            temperature=warmup_temperature,
            top_p=warmup_top_p,
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
        )

        self._warm_up(warmup_request)

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
                    seed=self.seed,
                )

                response, latency = self._generate_measured(
                    request
                )

                result = SamplingResult(
                    experiment=experiment,
                    temperature=temperature,
                    top_p=top_p,
                    repetition=repetition,
                    response=response.text,
                    latency_seconds=latency,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    seed=self.seed,
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

    def run_top_k(
        self,
        *,
        prompt: str,
        top_k_values: list[int],
        temperature: float,
        top_p: float,
        repetitions: int,
        on_result: ResultCallback | None = None,
    ) -> list[SamplingResult]:
        """Run native Top-k experiments."""

        if not top_k_values:
            raise ValueError(
                "Top-k values cannot be empty."
            )

        if any(
            type(value) is not int or value < 1
            for value in top_k_values
        ):
            raise ValueError(
                "Top-k values must be positive integers."
            )

        if (
            type(repetitions) is not int
            or repetitions < 1
        ):
            raise ValueError(
                "repetitions must be a positive integer."
            )

        warmup_request = GenerationRequest(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_values[0],
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
        )

        self._warm_up(warmup_request)

        results: list[SamplingResult] = []

        for top_k in top_k_values:
            for repetition in range(1, repetitions + 1):

                request = GenerationRequest(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=self.max_output_tokens,
                    seed=self.seed,
                )

                response, latency = self._generate_measured(
                    request
                )

                result = SamplingResult(
                    experiment="top_k",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition=repetition,
                    response=response.text,
                    latency_seconds=latency,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    seed=self.seed,
                )

                results.append(result)

                if on_result is not None:
                    on_result(result)

        return results