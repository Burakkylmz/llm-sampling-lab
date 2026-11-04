
from dataclasses import dataclass
from types import MappingProxyType

from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    ParameterRange,
    SupportStatus,
)


@dataclass(frozen=True)
class ModelMetadata:
    source: str
    last_verified: str | None = None
    live_verified: bool = False


@dataclass(frozen=True)
class ModelCatalogEntry:
    capabilities: ModelCapabilities
    metadata: ModelMetadata


SAMPLING_RANGES = MappingProxyType({
    "temperature": ParameterRange(
        minimum=0.0,
        maximum=2.0,
    ),
    "top_p": ParameterRange(
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    ),
})


def _create_entry(
    provider: str,
    model: str,
) -> ModelCatalogEntry:

    return ModelCatalogEntry(
        capabilities=ModelCapabilities(
            provider=provider,
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.UNSUPPORTED,
            parameter_ranges=SAMPLING_RANGES,
        ),
        metadata=ModelMetadata(
            source="local_static_catalog",
            last_verified=None,
            live_verified=False,
        ),
    )


MODEL_CATALOG = {
    ("groq", "openai/gpt-oss-20b"): _create_entry(
        "groq", "openai/gpt-oss-20b"
    ),
    ("groq", "openai/gpt-oss-120b"): _create_entry(
        "groq", "openai/gpt-oss-120b"
    ),
    ("openai", "gpt-4.1"): _create_entry(
        "openai", "gpt-4.1"
    ),
    ("openai", "gpt-4.1-mini"): _create_entry(
        "openai", "gpt-4.1-mini"
    ),
}


def get_model_capabilities(
    provider: str,
    model: str,
) -> ModelCapabilities:

    entry = MODEL_CATALOG.get((provider, model))

    if entry is not None:
        return entry.capabilities

    return ModelCapabilities(
        provider=provider,
        model=model,
    )


def get_model_metadata(
    provider: str,
    model: str,
) -> ModelMetadata:

    entry = MODEL_CATALOG.get((provider, model))

    if entry is not None:
        return entry.metadata

    return ModelMetadata(
        source="not_in_catalog",
    )


def list_catalog_models(
    provider: str,
) -> list[str]:

    return sorted(
        model
        for provider_name, model in MODEL_CATALOG
        if provider_name == provider
    )