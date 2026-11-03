
from typing import Any

from groq import Groq

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.groq_client import create_groq_client
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


class GroqProvider(BaseProvider):
    """Groq implementation of the common provider interface."""

    name = "groq"

    VERIFIED_MODELS = frozenset({
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    })

    def __init__(
        self,
        client: Groq | None = None,
    ) -> None:
        self._client = client
    
    def discover_models(self) -> list[DiscoveredModel]:
        """Retrieve models from the live Groq API."""

        if self._client is not None:
            response = self._client.models.list()
        else:
            with create_groq_client() as client:
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
        """Execute a validated Groq generation request."""

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

        # GPT-OSS-specific reasoning configuration.
        if request.model in self.VERIFIED_MODELS:
            parameters["reasoning_effort"] = "low"
            parameters["include_reasoning"] = False

        if self._client is not None:
            completion = (
                self._client.chat.completions.create(
                    **parameters
                )
            )
        else:
            with create_groq_client() as client:
                completion = (
                    client.chat.completions.create(
                        **parameters
                    )
                )

        if not completion.choices:
            raise ValueError("Groq returned no choices.")

        choice = completion.choices[0]
        content = choice.message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq returned an empty response.")

        usage = completion.usage

        return GenerationResponse(
            provider=self.name,
            model=request.model,
            text=content,
            finish_reason=choice.finish_reason,
            input_tokens=(
                getattr(usage, "prompt_tokens", None)
            ),
            output_tokens=(
                getattr(usage, "completion_tokens", None)
            ),
        )
