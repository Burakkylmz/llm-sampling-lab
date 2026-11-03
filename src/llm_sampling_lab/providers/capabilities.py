
import math

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParameterRange:
    """Allowed range for a sampling parameter."""

    minimum: float
    maximum: float
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    integer_only: bool = False

    def validate(
        self,
        name: str,
        value: object,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        if self.integer_only and type(value) is not int:
            raise ValueError(
                f"{name} must be an integer."
            )

        lower_valid = (
            value >= self.minimum
            if self.minimum_inclusive
            else value > self.minimum
        )

        upper_valid = (
            value <= self.maximum
            if self.maximum_inclusive
            else value < self.maximum
        )

        if not lower_valid or not upper_valid:
            left = "[" if self.minimum_inclusive else "("
            right = "]" if self.maximum_inclusive else ")"

            raise ValueError(
                f"{name} must be in "
                f"{left}{self.minimum}, {self.maximum}{right}. "
                f"Received: {value}"
            )


@dataclass(frozen=True)
class ModelCapabilities:
    """Sampling support and validation rules for a model."""

    provider: str
    model: str

    temperature: SupportStatus = SupportStatus.UNKNOWN
    top_p: SupportStatus = SupportStatus.UNKNOWN
    top_k: SupportStatus = SupportStatus.UNKNOWN

    parameter_ranges: Mapping[str, ParameterRange] = field(
        default_factory=dict
    )

    def validate(
        self,
        **parameters: object,
    ) -> None:
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

            parameter_range = self.parameter_ranges.get(name)

            if parameter_range is not None:
                parameter_range.validate(name, value)