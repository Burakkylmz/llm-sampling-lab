
import pytest

from llm_sampling_lab.config import (
    load_settings,
    validate_settings,
)


def test_valid_configuration():
    """Valid configuration should pass validation."""

    settings = load_settings()

    validate_settings(settings)


def test_invalid_groq_model_type():
    """Invalid Groq model type should raise ValueError."""

    settings = load_settings()
    settings["groq"]["model"] = 123

    with pytest.raises(
        ValueError,
        match="groq.model must be a non-empty string",
    ):
        validate_settings(settings)


@pytest.mark.parametrize(
    "section, field, invalid_value, expected_error",
    [
        ("temperature", "values", -1, "temperature must be"),
        ("temperature", "values", 2.5, "temperature must be"),
        ("top_p", "values", 0, "top_p must be"),
        ("top_p", "values", 1.5, "top_p must be"),
    ],
)
def test_invalid_sampling_parameters(
    section,
    field,
    invalid_value,
    expected_error,
):
    """Invalid sampling values should raise ValueError."""

    settings = load_settings()

    settings[section][field] = [invalid_value]

    with pytest.raises(ValueError, match=expected_error):
        validate_settings(settings)


@pytest.mark.parametrize(
    "section",
    [
        "groq",
        "local_sampling",
    ],
)
def test_missing_required_section(section):
    """Missing configuration sections should raise ValueError."""

    settings = load_settings()

    del settings[section]

    with pytest.raises(
        ValueError,
        match="Missing required configuration section",
    ):
        validate_settings(settings)


def test_invalid_configuration_section_type():
    """Configuration sections must be TOML tables."""

    settings = load_settings()

    settings["groq"] = "invalid"

    with pytest.raises(
        ValueError,
        match="must be a table",
    ):
        validate_settings(settings)



@pytest.mark.parametrize(
    "section, field",
    [
        ("groq", "model"),
        ("experiments", "repetitions"),
        ("temperature", "values"),
        ("local_sampling", "top_p_values"),
    ],
)
def test_missing_required_field(section, field):
    """Missing configuration fields should raise ValueError."""

    settings = load_settings()

    del settings[section][field]

    expected_error = (
        f"Missing required configuration field: "
        f"{section}.{field}"
    )

    with pytest.raises(
        ValueError,
        match=expected_error.replace(".", r"\."),
    ):
        validate_settings(settings)


