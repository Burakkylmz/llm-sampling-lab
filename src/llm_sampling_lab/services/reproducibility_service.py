
from collections import Counter, defaultdict
from typing import Any

from llm_sampling_lab.domain.models import SamplingResult


PARAMETER_FIELDS = {
    "temperature": "temperature",
    "top-p": "top_p",
    "top-k": "top_k",
}


def summarize_reproducibility(
    results: list[SamplingResult],
    strategy: str,
) -> list[dict[str, Any]]:
    """Analyze exact-text agreement across repetitions."""

    if strategy not in PARAMETER_FIELDS:
        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
        )

    field = PARAMETER_FIELDS[strategy]
    grouped = defaultdict(list)

    for result in results:
        grouped[getattr(result, field)].append(result)

    summary = []

    for parameter in sorted(grouped):
        group = grouped[parameter]

        responses = [
            result.response
            for result in group
        ]

        frequencies = Counter(responses)

        total = len(responses)
        most_common = frequencies.most_common(1)[0][1]

        seeds = {
            result.seed
            for result in group
        }

        seed = (
            next(iter(seeds))
            if len(seeds) == 1
            else "mixed"
        )

        summary.append({
            "Parameter": parameter,
            "Repetitions": total,
            "Unique Responses": len(frequencies),
            "Exact Match": (
                len(frequencies) == 1 and total >= 2
            ),
            "Most Common Ratio": round(
                most_common / total,
                3,
            ),
            "Seed": seed,
        })

    return summary