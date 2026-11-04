
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from llm_sampling_lab.domain.models import GenerationRequest
from llm_sampling_lab.providers.ollama_provider import OllamaProvider


def main() -> None:
    provider = OllamaProvider()

    prompt = (
        "Write one creative slogan for an electric car. "
        "Return only the slogan."
    )

    results = []

    for top_k in [1, 20, 40]:
        for repetition in range(1, 4):
            request = GenerationRequest(
                model="llama3.2:1b",
                prompt=prompt,
                temperature=0.8,
                top_p=0.9,
                top_k=top_k,
                max_output_tokens=80,
            )

            start = perf_counter()
            response = provider.generate(request)
            latency = perf_counter() - start

            result = {
                "top_k": top_k,
                "repetition": repetition,
                "latency_seconds": round(latency, 3),
                "response": asdict(response),
            }

            results.append(result)

            print(f"\nTop-k: {top_k}")
            print(f"Repetition: {repetition}")
            print(f"Latency: {latency:.2f}s")
            print(f"Response: {response.text}")

    output_dir = Path("results/ollama_top_k")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "experiment.json"

    output_file.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()