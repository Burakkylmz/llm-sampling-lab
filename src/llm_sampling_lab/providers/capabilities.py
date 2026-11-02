
from dataclasses import dataclass
from enum import Enum


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model: str

    temperature: SupportStatus = SupportStatus.UNKNOWN
    top_p: SupportStatus = SupportStatus.UNKNOWN
    top_k: SupportStatus = SupportStatus.UNKNOWN

    def validate(self, **parameters: object) -> None:
        allowed = {"temperature", "top_p", "top_k"}

        for name, value in parameters.items():
            if name not in allowed:
                raise ValueError(
                    f"Unknown parameter: {name}"
                )

            if value is None:
                continue

            status = getattr(self, name)

            if status != SupportStatus.SUPPORTED:
                raise ValueError(
                    f"{self.provider}/{self.model}: "
                    f"{name} is {status.value}"
                )