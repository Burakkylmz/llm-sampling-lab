
import sys
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli


@pytest.fixture
def mock_configuration(monkeypatch):
    """Provide valid configuration without reading TOML."""

    settings = {
        "groq": {"model": "test-model"},
        "experiments": {"repetitions": 5},
    }

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        cli,
        "validate_settings",
        lambda settings: None,
    )


def test_cli_without_api_check(
    monkeypatch,
    mock_configuration,
):
    """Normal CLI execution must not call the API."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab"],
    )

    mock_check_api = MagicMock()

    monkeypatch.setattr(
        cli,
        "check_api",
        mock_check_api,
    )

    cli.main()

    mock_check_api.assert_not_called()


def test_cli_with_api_check(
    monkeypatch,
    mock_configuration,
):
    """The --check-api flag must trigger API validation."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--check-api"],
    )

    mock_check_api = MagicMock()

    monkeypatch.setattr(
        cli,
        "check_api",
        mock_check_api,
    )

    cli.main()

    mock_check_api.assert_called_once_with("test-model")


def test_cli_api_check_failure(
    monkeypatch,
    mock_configuration,
    capsys,
):
    """An API response error must exit with code 1."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--check-api"],
    )

    mock_check_api = MagicMock(
        side_effect=ValueError(
            "Simulated API response failure"
        )
    )

    monkeypatch.setattr(
        cli,
        "check_api",
        mock_check_api,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1

    captured = capsys.readouterr()

    assert (
        "Simulated API response failure"
        in captured.err
    )



@pytest.mark.parametrize(
    "extra_args, expected_repetitions",
    [
        ([], 5),
        (["--repetitions", "1"], 1),
    ],
)
def test_temperature_experiment_cli(
    monkeypatch,
    mock_configuration,
    capsys,
    extra_args,
    expected_repetitions,
):
    """CLI should use configured or overridden repetitions."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--experiment",
            "temperature",
            *extra_args,
        ],
    )

    mock_runner = MagicMock()

    monkeypatch.setattr(
        cli,
        "run_temperature_cli",
        mock_runner,
    )

    cli.main()

    mock_runner.assert_called_once()

    kwargs = mock_runner.call_args.kwargs


    captured = capsys.readouterr()

    assert (
        f"Repetitions: {expected_repetitions}"
        in captured.out
    )


    assert kwargs["repetitions"] == expected_repetitions
    assert kwargs["settings"]["groq"]["model"] == "test-model"



@pytest.mark.parametrize(
    "extra_args, expected_repetitions",
    [
        ([], 5),
        (["--repetitions", "1"], 1),
    ],
)
def test_top_p_experiment_cli(
    monkeypatch,
    mock_configuration,
    capsys,
    extra_args,
    expected_repetitions,
):
    """Verify top-p experiment CLI routing and repetitions."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--experiment",
            "top-p",
            *extra_args,
        ],
    )

    mock_runner = MagicMock()

    monkeypatch.setattr(
        cli,
        "run_top_p_cli",
        mock_runner,
    )

    # Execute CLI
    cli.main()

    # Verify experiment runner invocation
    mock_runner.assert_called_once()

    kwargs = mock_runner.call_args.kwargs

    assert kwargs["repetitions"] == expected_repetitions
    assert kwargs["settings"]["groq"]["model"] == "test-model"

    # Verify CLI output
    captured = capsys.readouterr()

    assert (
        f"Repetitions: {expected_repetitions}"
        in captured.out
    )
