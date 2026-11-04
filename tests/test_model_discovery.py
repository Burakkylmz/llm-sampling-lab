
import sys

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli
from llm_sampling_lab.providers.groq_provider import GroqProvider
from llm_sampling_lab.providers.openai_provider import OpenAIProvider
from llm_sampling_lab.providers.model_discovery import (
    DiscoveredModel,
    parse_model_response,
)


def test_parse_groq_response():
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="openai/gpt-oss-20b",
                owned_by="OpenAI",
                active=True,
            )
        ]
    )

    models = parse_model_response("groq", response)

    assert len(models) == 1
    assert models[0].model_id == "openai/gpt-oss-20b"
    assert models[0].active is True


def test_parse_openai_response():
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="gpt-4.1-mini",
                owned_by="openai",
            )
        ]
    )

    models = parse_model_response("openai", response)

    assert len(models) == 1
    assert models[0].active is None


def test_groq_live_discovery():
    client = MagicMock()

    client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="openai/gpt-oss-20b",
                owned_by="OpenAI",
                active=True,
            )
        ]
    )

    provider = GroqProvider(client=client)
    models = provider.discover_models()

    client.models.list.assert_called_once_with()
    assert len(models) == 1


def test_openai_live_discovery():
    client = MagicMock()

    client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                id="gpt-4.1-mini",
                owned_by="openai",
            )
        ]
    )

    provider = OpenAIProvider(client=client)
    models = provider.discover_models()

    client.models.list.assert_called_once_with()
    assert models[0].provider == "openai"


def test_live_discovery_cli_without_model(
    monkeypatch,
    capsys,
):
    class FakeDiscoverer:
        def discover_models(self):
            return [
                DiscoveredModel(
                    provider="openai",
                    model_id="example-model",
                )
            ]

    monkeypatch.setattr(
        cli,
        "provider_session",
        lambda name: nullcontext(FakeDiscoverer()),
    )

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: {
            "groq": {"model": "openai/gpt-oss-20b"},
            "experiments": {"repetitions": 2},
        },
    )

    monkeypatch.setattr(
        cli,
        "validate_settings",
        lambda settings: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--provider", "openai",
            "--discover-models",
        ],
    )

    cli.main()

    output = capsys.readouterr().out

    assert "example-model" in output
    assert "Source: Live provider API" in output


def test_discovery_rejects_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--discover-models",
            "--model", "gpt-4.1-mini",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2


def test_discovery_and_local_list_are_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--list-models",
            "--discover-models",
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2