
import os
from typing import Any

from openai import OpenAI

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.providers.model_catalog import (
    get_model_capabilities,
)
from llm_sampling_lab.providers.model_discovery import (
    DiscoveredModel,
    parse_model_response,
)


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions provider adapter."""

    name = "openai"

    SAMPLING_MODELS = frozenset({
        "gpt-4.1",
        "gpt-4.1-mini",
    })

    def __init__(
        self,
        client: OpenAI | None = None,
    ) -> None:
        self._client = client

    def discover_models(self) -> list[DiscoveredModel]:
        """Retrieve models from the live OpenAI API."""

        if self._client is not None:
            response = self._client.models.list()
        else:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing or empty."
                )

            with OpenAI(
                api_key=api_key,
                timeout=30.0,
                max_retries=0,
            ) as client:
                response = client.models.list()

        return parse_model_response(
            provider=self.name,
            response=response,
        )

    def get_capabilities(
        self,
        model: str,
    ) -> ModelCapabilities:
        return get_model_capabilities(
            provider=self.name,
            model=model,
        )

    def _generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        parameters: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            "max_completion_tokens": request.max_output_tokens,
        }

        if request.temperature is not None:
            parameters["temperature"] = request.temperature

        if request.top_p is not None:
            parameters["top_p"] = request.top_p

        client = self._client

        if client is None:
            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise ValueError(
                    "OPENAI_API_KEY is missing or empty."
                )

            client = OpenAI(
                timeout=30.0,
                max_retries=0,
            )

        completion = client.chat.completions.create(
            **parameters
        )

        if not completion.choices:
            raise ValueError("OpenAI returned no choices.")

        choice = completion.choices[0]
        content = choice.message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI returned an empty response.")

        usage = completion.usage

        return GenerationResponse(
            provider=self.name,
            model=request.model,
            text=content,
            finish_reason=choice.finish_reason,
            input_tokens=(
                getattr(usage, "prompt_tokens", None)
                if usage is not None
                else None
            ),
            output_tokens=(
                getattr(usage, "completion_tokens", None)
                if usage is not None
                else None
            ),
        )