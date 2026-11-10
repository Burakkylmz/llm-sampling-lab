
import os

from ollama import Client

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
    SAMPLING_RANGES,
)
from llm_sampling_lab.providers.model_discovery import (
    DiscoveredModel,
)


class OllamaProvider(BaseProvider):
    """Adapter for locally hosted Ollama models."""

    name = "ollama"
    supports_seed = True

    def __init__(
        self,
        client: Client | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else Client(
                host=os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
            )
        )

    def discover_models(self) -> list[DiscoveredModel]:
        """List models installed on the local Ollama server."""

        response = self._client.list()
        models = []

        for item in response.models:
            if not item.model:
                continue

            models.append(
                DiscoveredModel(
                    provider=self.name,
                    model_id=item.model,
                )
            )

        return sorted(
            models,
            key=lambda item: item.model_id,
        )

    def get_capabilities(
        self,
        model: str,
    ) -> ModelCapabilities:
        """Check local installation and generation capability."""

        models = self.discover_models()

        installed = {
            item.model_id
            for item in models
        }

        # Ollama permits omitted ':latest' in model names.
        resolved_name = (
            model
            if ":" in model
            else f"{model}:latest"
        )

        if resolved_name not in installed:
            return ModelCapabilities(
                provider=self.name,
                model=model,
            )

        details = self._client.show(model)
        features = getattr(details, "capabilities", None)

        if not features:
            status = SupportStatus.UNKNOWN
        elif "completion" in features:
            status = SupportStatus.SUPPORTED
        else:
            status = SupportStatus.UNSUPPORTED

        return ModelCapabilities(
            provider=self.name,
            model=model,
            temperature=status,
            top_p=status,
            top_k=status,
            parameter_ranges=SAMPLING_RANGES,
        )

    def _generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate text with Ollama's native sampling."""

        options = {
            "num_predict": request.max_output_tokens,
        }

        if request.temperature is not None:
            options["temperature"] = request.temperature

        if request.top_p is not None:
            options["top_p"] = request.top_p

        if request.top_k is not None:
            options["top_k"] = request.top_k

        if request.seed is not None:
            options["seed"] = request.seed

        response = self._client.chat(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            options=options,
            stream=False,
        )

        content = response.message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Ollama returned an empty response."
            )

        return GenerationResponse(
            provider=self.name,
            model=request.model,
            text=content,
            finish_reason=getattr(
                response, "done_reason", None
            ),
            input_tokens=getattr(
                response, "prompt_eval_count", None
            ),
            output_tokens=getattr(
                response, "eval_count", None
            ),
        )