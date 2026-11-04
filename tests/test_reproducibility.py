
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    GenerationResponse,
)
from llm_sampling_lab.providers.base import BaseProvider
from llm_sampling_lab.providers.capabilities import (
    ModelCapabilities,
    SupportStatus,
)
from llm_sampling_lab.providers.groq_provider import GroqProvider
from llm_sampling_lab.providers.ollama_provider import OllamaProvider
from llm_sampling_lab.results import ExperimentRecorder
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)
from llm_sampling_lab.services.reproducibility_service import (
    summarize_reproducibility,
)


class FakeOllamaProvider(BaseProvider):
    name = "ollama"
    supports_seed = True

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

    def _generate(self, request):
        self.requests.append(request)

        return GenerationResponse(
            provider="ollama",
            model=request.model,
            text=f"Seed {request.seed} / Top-k {request.top_k}",
            input_tokens=10,
            output_tokens=5,
        )


def test_warmup_excluded_from_results():
    provider = FakeOllamaProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        warmup_requests=1,
        seed=42,
    )

    recorded = []

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
        on_result=recorded.append,
    )

    assert len(provider.requests) == 5
    assert len(results) == 4
    assert len(recorded) == 4

    assert provider.requests[0].top_k == 1
    assert all(item.seed == 42 for item in results)


def test_unsupported_provider_rejects_seed():
    provider = GroqProvider(client=MagicMock())

    with pytest.raises(ValueError, match="seed"):
        ExperimentService(
            provider=provider,
            model="openai/gpt-oss-20b",
            seed=42,
        )


def test_ollama_forwards_seed():
    client = MagicMock()

    client.list.return_value = SimpleNamespace(
        models=[
            SimpleNamespace(model="llama3.2:1b")
        ]
    )

    client.show.return_value = SimpleNamespace(
        capabilities=["completion"]
    )

    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="Hello"),
        done_reason="stop",
        prompt_eval_count=10,
        eval_count=5,
    )

    provider = OllamaProvider(client=client)

    provider.generate(
        GenerationRequest(
            model="llama3.2:1b",
            prompt="Hello",
            temperature=0.8,
            top_p=0.9,
            top_k=20,
            seed=42,
        )
    )

    options = client.chat.call_args.kwargs["options"]

    assert options["seed"] == 42
    assert options["top_k"] == 20


def test_seed_metadata_persisted(tmp_path):
    provider = FakeOllamaProvider()

    recorder = ExperimentRecorder(
        model="test-model",
        provider="ollama",
        prompt="Hello",
        experiment_type="top_k",
        parameter_values=[20],
        fixed_parameter="temperature",
        fixed_value=0.8,
        extra_fixed_parameters={"top_p": 0.9},
        repetitions=2,
        warmup_requests=1,
        seed=42,
        output_dir=tmp_path,
    )

    service = ExperimentService(
        provider=provider,
        model="test-model",
        warmup_requests=1,
        seed=42,
    )

    service.run_top_k(
        prompt="Hello",
        top_k_values=[20],
        temperature=0.8,
        top_p=0.9,
        repetitions=2,
        on_result=recorder.record,
    )

    recorder.complete()

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert len(provider.requests) == 3
    assert len(data["results"]) == 2
    assert data["parameters"]["seed"] == 42
    assert data["parameters"]["warmup_requests"] == 1
    assert all(
        item["seed"] == 42
        for item in data["results"]
    )


def test_reproducibility_exact_match():
    provider = FakeOllamaProvider()

    service = ExperimentService(
        provider=provider,
        model="test-model",
        seed=42,
    )

    results = service.run_top_k(
        prompt="Hello",
        top_k_values=[1, 20],
        temperature=0.8,
        top_p=0.9,
        repetitions=3,
    )

    summary = summarize_reproducibility(
        results,
        "top-k",
    )

    assert len(summary) == 2
    assert all(item["Exact Match"] for item in summary)
    assert all(item["Seed"] == 42 for item in summary)


def test_invalid_seed_is_rejected():
    provider = FakeOllamaProvider()

    with pytest.raises(ValueError, match="seed"):
        ExperimentService(
            provider=provider,
            model="test-model",
            seed=-1,
        )
