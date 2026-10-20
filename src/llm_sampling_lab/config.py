
import math
import tomllib

from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------

def load_settings(
    config_path: str | Path = "settings.toml",
) -> dict[str, Any]:
    """Load application settings from a TOML file."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}"
        )

    with path.open("rb") as file:
        settings = tomllib.load(file)

    return settings


# ---------------------------------------------------------
# Common Validation Helpers
# ---------------------------------------------------------

def _check_range(
    name: str,
    value: Any,
    minimum: float,
    maximum: float,
) -> None:
    """Validate a numeric value within an inclusive range."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}. "
            f"Received: {value!r}"
        )


def _check_non_empty_list(
    name: str,
    values: Any,
) -> None:
    """Validate that a value is a non-empty list."""

    if not isinstance(values, list) or not values:
        raise ValueError(
            f"{name} must be a non-empty list."
        )


# ---------------------------------------------------------
# Configuration Structure Validation
# ---------------------------------------------------------

def _validate_required_sections(
    settings: dict[str, Any],
) -> None:
    """Validate required TOML sections."""

    required_sections = (
        "groq",
        "experiments",
        "temperature",
        "top_p",
        "local_sampling",
    )

    for section in required_sections:

        if section not in settings:
            raise ValueError(
                f"Missing required configuration section: [{section}]"
            )

        if not isinstance(settings[section], dict):
            raise ValueError(
                f"Configuration section [{section}] must be a table."
            )


def _validate_required_fields(
    settings: dict[str, Any],
) -> None:
    """Validate required fields in TOML sections."""

    required_fields = {
        "groq": ("model",),
        "experiments": ("repetitions",),
        "temperature": ("values", "top_p"),
        "top_p": ("values", "temperature"),
        "local_sampling": (
            "top_k_values",
            "top_p_values",
            "temperature_values",
        ),
    }

    for section, fields in required_fields.items():

        for field in fields:

            if field not in settings[section]:
                raise ValueError(
                    f"Missing required configuration field: "
                    f"{section}.{field}"
                )


# ---------------------------------------------------------
# Groq Configuration Validation
# ---------------------------------------------------------

def _validate_groq_settings(
    settings: dict[str, Any],
) -> None:
    """Validate Groq model configuration."""

    model = settings["groq"]["model"]

    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "groq.model must be a non-empty string."
        )


# ---------------------------------------------------------
# Experiment Configuration Validation
# ---------------------------------------------------------

def _validate_experiment_settings(
    settings: dict[str, Any],
) -> None:
    """Validate general experiment configuration."""

    repetitions = settings["experiments"]["repetitions"]

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError(
            "repetitions must be a positive integer."
        )


# ---------------------------------------------------------
# Groq API Sampling Validation
# ---------------------------------------------------------

def _validate_api_sampling_settings(
    settings: dict[str, Any],
) -> None:
    """Validate temperature and top-p experiment settings."""

    # Temperature experiments
    temperature_values = settings["temperature"]["values"]

    _check_non_empty_list(
        "temperature.values",
        temperature_values,
    )

    for value in temperature_values:
        _check_range(
            "temperature",
            value,
            0.0,
            2.0,
        )

    _check_range(
        "temperature.top_p",
        settings["temperature"]["top_p"],
        0.000001,
        1.0,
    )

    # Top-p experiments
    top_p_values = settings["top_p"]["values"]

    _check_non_empty_list(
        "top_p.values",
        top_p_values,
    )

    for value in top_p_values:
        _check_range(
            "top_p",
            value,
            0.000001,
            1.0,
        )

    _check_range(
        "top_p.temperature",
        settings["top_p"]["temperature"],
        0.0,
        2.0,
    )


# ---------------------------------------------------------
# Local Sampling Validation
# ---------------------------------------------------------

def _validate_local_sampling_settings(
    settings: dict[str, Any],
) -> None:
    """Validate local sampling simulator settings."""

    local_settings = settings["local_sampling"]

    # Top-k
    top_k_values = local_settings["top_k_values"]

    _check_non_empty_list(
        "top_k_values",
        top_k_values,
    )

    for k in top_k_values:

        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k < 1
        ):
            raise ValueError(
                f"top_k must be a positive integer. Received: {k!r}"
            )

    # Temperature
    temperature_values = local_settings["temperature_values"]

    _check_non_empty_list(
        "temperature_values",
        temperature_values,
    )

    for value in temperature_values:
        _check_range(
            "local_sampling.temperature",
            value,
            0.000001,
            2.0,
        )

    # Top-p
    top_p_values = local_settings["top_p_values"]

    _check_non_empty_list(
        "top_p_values",
        top_p_values,
    )

    for value in top_p_values:
        _check_range(
            "local_sampling.top_p",
            value,
            0.000001,
            1.0,
        )


# ---------------------------------------------------------
# Main Validation Entry Point
# ---------------------------------------------------------

def validate_settings(
    settings: dict[str, Any],
) -> None:
    """Validate the complete application configuration."""

    _validate_required_sections(settings)

    _validate_required_fields(settings)

    _validate_groq_settings(settings)

    _validate_experiment_settings(settings)

    _validate_api_sampling_settings(settings)

    _validate_local_sampling_settings(settings)
