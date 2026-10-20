
import json
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli, experiments


@pytest.fixture
def experiment_settings():
    """Provide deterministic sampling configuration."""

    return {
        "groq": {
            "model": "test-model",
        },
        "temperature": {
            "values": [0.1, 0.5],
            "top_p": 1.0,
        },
        "top_p": {
            "values": [0.3, 0.7],
            "temperature": 0.7,
        },
    }


@pytest.mark.parametrize(
    "runner_name, directory",
    [
        ("run_temperature_cli", "temperature"),
        ("run_top_p_cli", "top_p"),
    ],
)
def test_experiment_json_persistence(
    monkeypatch,
    tmp_path,
    experiment_settings,
    runner_name,
    directory,
):
    """Both experiments should save completed JSON reports."""

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        cli,
        "create_groq_client",
        lambda: MagicMock(),
    )

    mock_generate = MagicMock(
        side_effect=[
            "Slogan A",
            "Slogan A",
            "Slogan B",
            "Slogan C",
        ]
    )

    monkeypatch.setattr(
        experiments,
        "generate_response",
        mock_generate,
    )

    runner = getattr(cli, runner_name)

    runner(
        settings=experiment_settings,
        repetitions=2,
    )

    report_files = list(
        (tmp_path / "results" / directory).glob("*.json")
    )

    assert len(report_files) == 1
    assert mock_generate.call_count == 4

    data = json.loads(
        report_files[0].read_text(encoding="utf-8")
    )

    assert data["status"] == "completed"
    assert data["experiment_type"] == directory
    assert len(data["results"]) == 4

    assert data["summary"][0]["unique_response_ratio"] == 0.5
    assert data["summary"][1]["unique_response_ratio"] == 1.0


def test_api_failure_preserves_partial_results(
    monkeypatch,
    tmp_path,
    experiment_settings,
):
    """Earlier successful responses must survive an API failure."""

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        cli,
        "create_groq_client",
        lambda: MagicMock(),
    )

    mock_generate = MagicMock(
        side_effect=[
            "Saved response",
            RuntimeError("Simulated API failure"),
        ]
    )

    monkeypatch.setattr(
        experiments,
        "generate_response",
        mock_generate,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated API failure",
    ):
        cli.run_temperature_cli(
            settings=experiment_settings,
            repetitions=2,
        )

    report_files = list(
        (tmp_path / "results" / "temperature").glob("*.json")
    )

    assert len(report_files) == 1

    data = json.loads(
        report_files[0].read_text(encoding="utf-8")
    )

    assert data["status"] == "failed"
    assert data["error_type"] == "RuntimeError"

    assert len(data["results"]) == 1
    assert data["results"][0]["response"] == "Saved response"
