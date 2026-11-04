
import math
from abc import ABC, abstractmethod

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
)


class BaseProvider(ABC):
    supports_seed = False

    @abstractmethod
    def get_capabilities(
        self, model: str
    ) -> ModelCapabilities:
        pass

    def generate(
        self, request: GenerationRequest
    ) -> GenerationResponse:

        if not request.model.strip():
            raise ValueError("Model cannot be empty")

        if not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if (
            type(request.max_output_tokens) is not int
            or request.max_output_tokens < 1
        ):
            raise ValueError("Invalid max_output_tokens")

        for name, value in (
            ("temperature", request.temperature),
            ("top_p", request.top_p),
        ):
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Invalid {name}")

        if request.temperature is not None:
            if request.temperature < 0:
                raise ValueError("Invalid temperature")

        if request.top_p is not None:
            if not 0 < request.top_p <= 1:
                raise ValueError("Invalid top_p")

        if request.top_k is not None:
            if (
                type(request.top_k) is not int
                or request.top_k < 1
            ):
                raise ValueError("Invalid top_k")

        if request.seed is not None:
            if (
                type(request.seed) is not int
                or request.seed < 0
            ):
                raise ValueError(
                    "Invalid seed: expected a non-negative integer."
                )

            if not self.supports_seed:
                raise ValueError(
                    f"{type(self).__name__} does not support seed."
                )
                
        capabilities = self.get_capabilities(
            request.model
        )

        capabilities.validate(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        return self._generate(request)

    @abstractmethod
    def _generate(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        pass