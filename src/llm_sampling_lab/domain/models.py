
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int = 256
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResponse:
    provider: str
    model: str
    text: str

    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class SamplingResult:
    """Provider-independent sampling experiment result."""

    experiment: str
    temperature: float
    top_p: float
    repetition: int
    response: str
    top_k: int | None = None

    # Benchmark measurements
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    seed: int | None = None