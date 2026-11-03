
from groq import Groq
from ollama import Client as OllamaClient
from openai import OpenAI

from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.groq_provider import GroqProvider
from llm_sampling_lab.providers.ollama_provider import OllamaProvider
from llm_sampling_lab.providers.openai_provider import OpenAIProvider
from llm_sampling_lab.providers.registry import ProviderRegistry


def build_provider_registry(
    groq_client: Groq | None = None,
    openai_client: OpenAI | None = None,
    ollama_client: OllamaClient | None = None,
) -> ProviderRegistry:
    """Register configured provider factories."""

    registry = ProviderRegistry()

    registry.register(
        "groq",
        lambda: GroqProvider(client=groq_client),
    )

    registry.register(
        "openai",
        lambda: OpenAIProvider(client=openai_client),
    )

    registry.register(
        "ollama",
        lambda: OllamaProvider(client=ollama_client),
    )

    return registry


def create_provider(
    name: str,
    groq_client: Groq | None = None,
    openai_client: OpenAI | None = None,
    ollama_client: OllamaClient | None = None,
) -> BaseProvider:
    """Create a provider via the central registry."""

    registry = build_provider_registry(
        groq_client=groq_client,
        openai_client=openai_client,
        ollama_client=ollama_client,
    )

    return registry.create(name)