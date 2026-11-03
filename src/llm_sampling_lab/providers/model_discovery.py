
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DiscoveredModel:
    """Basic metadata returned by a live provider API."""

    provider: str
    model_id: str
    owned_by: str | None = None
    active: bool | None = None


@runtime_checkable
class SupportsModelDiscovery(Protocol):
    """Optional interface for providers supporting model listing."""

    def discover_models(self) -> list[DiscoveredModel]:
        ...


def parse_model_response(
    provider: str,
    response: object,
) -> list[DiscoveredModel]:
    """Normalize provider model-list responses."""

    data = getattr(response, "data", None)

    if not isinstance(data, list):
        raise ValueError(
            f"{provider}: invalid model-list response."
        )

    models = []

    for item in data:
        model_id = getattr(item, "id", None)

        if not isinstance(model_id, str) or not model_id:
            raise ValueError(
                f"{provider}: model without a valid ID."
            )

        owner = getattr(item, "owned_by", None)
        active = getattr(item, "active", None)

        models.append(
            DiscoveredModel(
                provider=provider,
                model_id=model_id,
                owned_by=(
                    owner if isinstance(owner, str) else None
                ),
                active=(
                    active if isinstance(active, bool) else None
                ),
            )
        )

    return sorted(models, key=lambda item: item.model_id)