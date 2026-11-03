
import sys

import pytest

from llm_sampling_lab import cli

from llm_sampling_lab.providers.model_catalog import (
    get_model_metadata,
    list_catalog_models,
)


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


def test_groq_model_listing(
    monkeypatch, mock_settings, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--list-models"],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "openai/gpt-oss-20b" in output
    assert "openai/gpt-oss-120b" in output


def test_openai_model_listing_without_api_key(
    monkeypatch, mock_settings, capsys
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--list-models",
        ],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "gpt-4.1-mini" in output
    assert "gpt-4.1" in output


def test_model_metadata():
    metadata = get_model_metadata(
        "openai", "gpt-4.1-mini"
    )

    assert metadata.source == "local_static_catalog"
    assert metadata.live_verified is False
    assert metadata.last_verified is None


def test_unknown_model_metadata():
    metadata = get_model_metadata(
        "openai", "unverified-model"
    )

    assert metadata.source == "not_in_catalog"


def test_catalog_models():
    models = list_catalog_models("groq")

    assert len(models) == 2
    assert models == sorted(models)