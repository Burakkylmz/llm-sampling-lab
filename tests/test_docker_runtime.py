"""Docker-related configuration and provider routing regression checks."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_sampling_lab import cli
from llm_sampling_lab.providers import ollama_provider


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("configured_host", "expected_host"),
    [
        (None, "http://127.0.0.1:11434"),
        ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
    ],
)
def test_ollama_provider_honors_configurable_host(
    monkeypatch, configured_host, expected_host
):
    hosts = []

    def fake_client(*, host):
        hosts.append(host)
        return SimpleNamespace()

    monkeypatch.setattr(ollama_provider, "Client", fake_client)
    if configured_host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", configured_host)

    provider = ollama_provider.OllamaProvider()

    assert provider is not None
    assert hosts == [expected_host]


@pytest.mark.parametrize(
    ("configured_host", "expected_host"),
    [
        (None, "http://127.0.0.1:11434"),
        ("http://host.docker.internal:11434", "http://host.docker.internal:11434"),
    ],
)
def test_ollama_cli_session_honors_configurable_host(
    monkeypatch, configured_host, expected_host
):
    hosts = []

    class FakeClient:
        def __init__(self, *, host, timeout):
            hosts.append((host, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    monkeypatch.setattr(
        cli, "create_provider",
        lambda name, ollama_client: (name, ollama_client),
    )

    if configured_host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", configured_host)

    with cli.provider_session("ollama") as (name, client):
        assert name == "ollama"
        assert isinstance(client, FakeClient)

    assert hosts == [(expected_host, 120.0)]


def test_streamlit_is_in_docker_runtime_and_compose_keeps_cli():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")

    assert "COPY streamlit_app.py ./streamlit_app.py" in dockerfile
    assert "COPY .streamlit/ ./.streamlit/" in dockerfile
    assert 'ENTRYPOINT ["sampling-lab"]' in dockerfile

    assert "  ui:" in compose
    assert "  sampling-lab:" in compose
    assert "    profiles: [\"cli\"]" in compose
    assert "127.0.0.1:" in compose
    assert "--server.address=0.0.0.0" in compose
    assert "./results:/app/results" in compose
    assert "./settings.toml:/app/settings.toml:ro" in compose
    assert "OPENAI_API_KEY:" in compose
    assert "GROQ_API_KEY:" in compose
    assert "OLLAMA_HOST:" in compose
