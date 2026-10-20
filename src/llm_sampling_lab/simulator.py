
from typing import Any
from pathlib import Path

import numpy as np

from llm_sampling_lab.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token_indices,
)

from llm_sampling_lab.visualization import (
    save_sampling_comparison,
)


# Educational token distribution
BASE_TOKENS = [
    "AI",
    "Machine",
    "Deep",
    "Neural",
    "Data",
]

BASE_PROBABILITIES = [
    0.40,
    0.25,
    0.15,
    0.12,
    0.08,
]


def run_simulation_cli(
    settings: dict[str, Any],
    strategy: str,
    samples: int = 10000,
    seed: int = 42,
) -> None:
    """Run offline token sampling simulations."""

    strategies = {
        "temperature": (
            "temperature_values",
            apply_temperature,
        ),
        "top-k": (
            "top_k_values",
            apply_top_k,
        ),
        "top-p": (
            "top_p_values",
            apply_top_p,
        ),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unsupported simulation strategy: {strategy}"
        )

    if samples < 1:
        raise ValueError("samples must be positive.")

    if seed < 0:
        raise ValueError("seed cannot be negative.")

    config_key, transform = strategies[strategy]

    parameter_values = settings["local_sampling"][config_key]

    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSimulation: {strategy}")
    print(f"Samples: {samples}")
    print(f"Base Seed: {seed}")

    for index, value in enumerate(parameter_values):

        # Transform the probability distribution
        distribution = transform(
            BASE_PROBABILITIES,
            value,
        )

        # Sample from the transformed distribution
        selected_indices = sample_token_indices(
            probabilities=distribution,
            n_samples=samples,
            seed=seed + index,
        )

        # Calculate empirical probabilities
        counts = np.bincount(
            selected_indices,
            minlength=len(BASE_TOKENS),
        )

        observed = counts / samples

        # Save visualization
        value_label = str(value).replace(".", "p")

        output_path = output_dir / (
            f"{strategy}_{value_label}.png"
        )

        saved_path = save_sampling_comparison(
            tokens=BASE_TOKENS,
            probabilities=distribution,
            samples=selected_indices,
            output_path=output_path,
        )

        print("\n" + "=" * 55)
        print(f"{strategy.upper()}: {value}")
        print("=" * 55)

        print(
            f"{'Token':<12}"
            f"{'Expected':>12}"
            f"{'Observed':>12}"
        )

        for token, expected, actual in zip(
            BASE_TOKENS,
            distribution,
            observed,
        ):
            print(
                f"{token:<12}"
                f"{expected:>12.4f}"
                f"{actual:>12.4f}"
            )

        print(f"\nVisualization: {saved_path.resolve()}")
