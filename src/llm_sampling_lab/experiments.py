
from collections.abc import Callable
from typing import Any, Literal

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
    SamplingResult,
)
from llm_sampling_lab.groq_client import generate_response
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


SamplingParameter = Literal["temperature", "top_p"]
ResultCallback = Callable[[SamplingResult], None]


class _LegacyGroqAdapter(BaseProvider):
    """
    Compatibility adapter for the existing Groq CLI.

    Preserves the original generate_response interface.
    This is not the model capability discovery system.
    """

    def __init__(
        self,
        client: Any,
        generator: Callable[..., str],
    ) -> None:
        self._client = client
        self._generator = generator

    def get_capabilities(
        self,
        model: str,
    ) -> ModelCapabilities:
        # Preserve the old CLI's parameter behavior.
        # Actual model capabilities are not inferred here.
        return ModelCapabilities(
            provider="groq-legacy",
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.UNSUPPORTED,
        )

    def _generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        text = self._generator(
            client=self._client,
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return GenerationResponse(
            provider="groq",
            model=request.model,
            text=text,
        )


def _run_experiment(
    client: Any,
    model: str,
    prompt: str,
    parameter: SamplingParameter,
    values: list[float],
    fixed_value: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:
    """Backward-compatible entry point."""

    provider = _LegacyGroqAdapter(
        client=client,
        generator=generate_response,
    )

    service = ExperimentService(
        provider=provider,
        model=model,
    )

    if parameter == "temperature":
        return service.run_temperature(
            prompt=prompt,
            temperatures=values,
            top_p=fixed_value,
            repetitions=repetitions,
            on_result=on_result,
        )

    if parameter == "top_p":
        return service.run_top_p(
            prompt=prompt,
            top_p_values=values,
            temperature=fixed_value,
            repetitions=repetitions,
            on_result=on_result,
        )

    raise ValueError(
        f"Unsupported experiment parameter: {parameter}"
    )


def run_temperature_experiment(
    client: Any,
    model: str,
    prompt: str,
    temperature_values: list[float],
    top_p: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:

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
    client: Any,
    model: str,
    prompt: str,
    top_p_values: list[float],
    temperature: float,
    repetitions: int,
    on_result: ResultCallback | None = None,
) -> list[SamplingResult]:

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