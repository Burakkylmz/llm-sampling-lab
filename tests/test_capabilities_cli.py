
import sys

import pytest

from llm_sampling_lab import cli


@pytest.fixture
def mock_settings(monkeypatch):
    settings = {
        "groq": {"model": "openai/gpt-oss-20b"},
        "experiments": {"repetitions": 2},
    }

    monkeypatch.setattr(
        cli, "load_settings", lambda: settings
    )
    monkeypatch.setattr(
        cli, "validate_settings", lambda s: None
    )


def test_groq_capabilities(
    monkeypatch, mock_settings, capsys
):
    monkeypatch.setattr(
        sys, "argv",
        ["sampling-lab", "--show-capabilities"],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "Provider: groq" in output
    assert "temperature: supported" in output
    assert "top_k: unsupported" in output


def test_openai_capabilities(
    monkeypatch, mock_settings, capsys
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        sys, "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--model", "gpt-4.1-mini",
            "--show-capabilities",
        ],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "Provider: openai" in output
    assert "Model: gpt-4.1-mini" in output
    assert "top_p: supported" in output


def test_unknown_model(
    monkeypatch, mock_settings, capsys
):
    monkeypatch.setattr(
        sys, "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--model", "unverified-model",
            "--show-capabilities",
        ],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "temperature: unknown" in output


def test_capabilities_mutually_exclusive(
    monkeypatch,
):
    monkeypatch.setattr(
        sys, "argv",
        [
            "sampling-lab",
            "--show-capabilities",
            "--experiment", "temperature",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2