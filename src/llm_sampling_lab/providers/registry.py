
from collections.abc import Callable

from llm_sampling_lab.providers.base import BaseProvider


class ProviderRegistry:

    def __init__(self) -> None:
        self._providers: dict[
            str, Callable[[], BaseProvider]
        ] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], BaseProvider],
    ) -> None:

        name = name.strip().lower()

        if not name:
            raise ValueError("Provider name cannot be empty")

        if name in self._providers:
            raise ValueError(
                f"Provider already registered: {name}"
            )

        self._providers[name] = factory

    def create(self, name: str) -> BaseProvider:
        name = name.strip().lower()

        try:
            factory = self._providers[name]
        except KeyError:
            raise ValueError(
                f"Unknown provider: {name}"
            ) from None

        return factory()

    def available(self) -> list[str]:
        return sorted(self._providers)
