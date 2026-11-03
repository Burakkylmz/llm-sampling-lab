
import json
import sys

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli
from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)


class FakeTopKProvider(BaseProvider):
    def __init__(self):
        self.requests = []

    def get_capabilities(self, model):
        return ModelCapabilities(
            provider="ollama",
            model=model,
            temperature=SupportStatus.SUPPORTED,
            top_p=SupportStatus.SUPPORTED,
            top_k=SupportStatus.SUPPORTED,
        )

    def _generate(self, request: GenerationRequest):
        self.requests.append(request)

        return GenerationResponse(
            provider="ollama",
            model=request.model,
            text=f"Response for {request.top_k}",
        )


def test_top_k_experiment():
    provider = FakeTopKProvider()
    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    results = service.run_top_k(
        prompt="Write a slogan",
        top_k_values=[1, 20, 40],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
    )

    assert len(results) == 6
    assert [r.top_k for r in results] == [
        1, 1, 20, 20, 40, 40,
    ]
    assert all(r.experiment == "top_k" for r in results)
    assert len(provider.requests) == 6


def test_invalid_top_k_values():
    provider = FakeTopKProvider()
    service = ExperimentService(
        provider=provider,
        model="test-model",
    )

    with pytest.raises(ValueError, match="Top-k"):
        service.run_top_k(
            prompt="Hello",
            top_k_values=[0],
            temperature=0.8,
            top_p=0.9,
            repetitions=2,
        )

    assert provider.requests == []


def test_top_k_json_persistence(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    provider = MagicMock()
    provider.generate.side_effect = lambda request: (
        GenerationResponse(
            provider="ollama",
            model=request.model,
            text=f"Response for {request.top_k}",
        )
    )

    monkeypatch.setattr(
        cli,
        "provider_session",
        lambda name: nullcontext(provider),
    )

    cli.run_top_k_cli(
        settings={},
        repetitions=2,
        provider_name="ollama",
        model="test-model",
    )

    files = list(
        (tmp_path / "results" / "top_k").glob("*.json")
    )

    assert len(files) == 1

    data = json.loads(
        files[0].read_text(encoding="utf-8")
    )

    assert data["status"] == "completed"
    assert data["experiment_type"] == "top_k"
    assert data["provider"] == "ollama"
    assert len(data["results"]) == 6
    assert len(data["summary"]) == 3
    assert data["parameters"]["fixed"] == {
        "temperature": 0.8,
        "top_p": 0.9,
    }


def test_top_k_rejects_other_providers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--experiment", "top-k"],
    )

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
