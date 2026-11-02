
import json
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli

from llm_sampling_lab.domain.models import (
    GenerationResponse,
)


@pytest.fixture
def experiment_settings():
    """Provide deterministic experiment settings."""

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



def mock_provider_for_cli(
    monkeypatch,
    mock_generate,
):
    """Mock provider creation through the central factory."""

    provider = MagicMock()

    def fake_generate(request):
        text = mock_generate()

        return GenerationResponse(
            provider="groq",
            model=request.model,
            text=text,
        )

    provider.generate.side_effect = fake_generate

    def fake_create_provider(
        name,
        groq_client=None,
    ):
        assert name == "groq"
        assert groq_client is not None

        return provider

    monkeypatch.setattr(
        cli,
        "create_provider",
        fake_create_provider,
    )

    return provider


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
    """Both experiments must persist completed JSON reports."""

    monkeypatch.chdir(tmp_path)

    # Avoid real API requests.
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

    provider = mock_provider_for_cli(
        monkeypatch,
        mock_generate,
    )

    runner = getattr(cli, runner_name)

    runner(
        settings=experiment_settings,
        repetitions=2,
    )

    # Verify provider execution.
    assert provider.generate.call_count == 4
    assert mock_generate.call_count == 4

    # Verify JSON file creation.
    report_files = list(
        (tmp_path / "results" / directory).glob(
            "*.json"
        )
    )

    assert len(report_files) == 1

    data = json.loads(
        report_files[0].read_text(
            encoding="utf-8"
        )
    )

    # Verify experiment metadata.
    assert data["status"] == "completed"
    assert data["experiment_type"] == directory
    assert len(data["results"]) == 4

    # Verify diversity metrics.
    assert (
        data["summary"][0]["unique_response_ratio"]
        == 0.5
    )

    assert (
        data["summary"][1]["unique_response_ratio"]
        == 1.0
    )

    # Verify result ordering.
    responses = [
        item["response"]
        for item in data["results"]
    ]

    assert responses == [
        "Slogan A",
        "Slogan A",
        "Slogan B",
        "Slogan C",
    ]


def test_api_failure_preserves_partial_results(
    monkeypatch,
    tmp_path,
    experiment_settings,
):
    """An API failure must preserve successful prior results."""

    monkeypatch.chdir(tmp_path)

    # Mock Groq client creation.
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

    provider = mock_provider_for_cli(
        monkeypatch,
        mock_generate,
    )

    # The second generation request fails.
    with pytest.raises(
        RuntimeError,
        match="Simulated API failure",
    ):
        cli.run_temperature_cli(
            settings=experiment_settings,
            repetitions=2,
        )

    assert provider.generate.call_count == 2

    # Verify the report still exists.
    report_files = list(
        (tmp_path / "results" / "temperature").glob(
            "*.json"
        )
    )

    assert len(report_files) == 1

    data = json.loads(
        report_files[0].read_text(
            encoding="utf-8"
        )
    )

    # Failure status must be recorded.
    assert data["status"] == "failed"

    assert data["error_type"] == "RuntimeError"

    # The successful response must not be lost.
    assert len(data["results"]) == 1

    assert (
        data["results"][0]["response"]
        == "Saved response"
    )