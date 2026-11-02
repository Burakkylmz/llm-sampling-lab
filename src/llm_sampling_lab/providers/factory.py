
from groq import Groq
from openai import OpenAI

from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.groq_provider import GroqProvider
from llm_sampling_lab.providers.openai_provider import OpenAIProvider
from llm_sampling_lab.providers.registry import ProviderRegistry


def build_provider_registry(
    groq_client: Groq | None = None,
    openai_client: OpenAI | None = None,
) -> ProviderRegistry:
    """Build the registry of available LLM providers."""

    registry = ProviderRegistry()

    registry.register(
        "groq",
        lambda: GroqProvider(client=groq_client),
    )

    registry.register(
        "openai",
        lambda: OpenAIProvider(client=openai_client),
    )

    return registry


def create_provider(
    name: str,
    groq_client: Groq | None = None,
    openai_client: OpenAI | None = None,
) -> BaseProvider:
    """Create a provider using the central registry."""

    registry = build_provider_registry(
        groq_client=groq_client,
        openai_client=openai_client,
    )

    return registry.create(name)