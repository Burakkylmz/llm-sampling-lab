
import json
import sys
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli
from llm_sampling_lab.domain.models import GenerationResponse


@pytest.fixture
def settings():
    return {
        "groq": {"model": "openai/gpt-oss-20b"},
        "experiments": {"repetitions": 2},
        "temperature": {
            "values": [0.5],
            "top_p": 1.0,
        },
        "top_p": {
            "values": [0.7],
            "temperature": 0.5,
        },
    }


def test_default_provider_is_groq(
    monkeypatch,
    settings,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--experiment", "temperature"],
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "validate_settings", lambda s: None)

    runner = MagicMock()
    monkeypatch.setattr(cli, "run_temperature_cli", runner)

    cli.main()

    kwargs = runner.call_args.kwargs

    assert kwargs["provider_name"] == "groq"
    assert kwargs["model"] == "openai/gpt-oss-20b"
    assert kwargs["repetitions"] == 2


def test_openai_provider_selection(
    monkeypatch,
    settings,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--model", "gpt-4.1-mini",
            "--experiment", "top-p",
        ],
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "validate_settings", lambda s: None)

    runner = MagicMock()
    monkeypatch.setattr(cli, "run_top_p_cli", runner)

    cli.main()

    kwargs = runner.call_args.kwargs

    assert kwargs["provider_name"] == "openai"
    assert kwargs["model"] == "gpt-4.1-mini"


def test_openai_requires_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--experiment", "temperature",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2


def test_openai_experiment_persistence(
    monkeypatch,
    tmp_path,
    settings,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

    client = MagicMock()
    monkeypatch.setattr(
        cli,
        "OpenAI",
        lambda **kwargs: client,
    )

    provider = MagicMock()
    provider.generate.return_value = GenerationResponse(
        provider="openai",
        model="gpt-4.1-mini",
        text="Test response",
    )

    def fake_create_provider(name, openai_client=None):
        assert name == "openai"
        assert openai_client is not None
        return provider

    monkeypatch.setattr(
        cli,
        "create_provider",
        fake_create_provider,
    )

    cli.run_temperature_cli(
        settings=settings,
        repetitions=2,
        provider_name="openai",
        model="gpt-4.1-mini",
    )

    assert provider.generate.call_count == 2

    reports = list(
        (tmp_path / "results" / "temperature").glob("*.json")
    )

    assert len(reports) == 1

    data = json.loads(
        reports[0].read_text(encoding="utf-8")
    )

    assert data["status"] == "completed"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4.1-mini"
    assert len(data["results"]) == 2