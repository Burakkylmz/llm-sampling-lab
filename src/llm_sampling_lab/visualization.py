
from pathlib import Path
from collections.abc import Sequence

import matplotlib

# Headless backend for Docker environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_sampling_comparison(
    tokens: Sequence[str],
    probabilities: Sequence[float],
    samples: Sequence[int],
    output_path: str | Path,
) -> Path:
    """Save theoretical vs observed token probabilities as PNG."""

    token_list = list(tokens)

    expected = np.asarray(probabilities, dtype=np.float64)
    sampled_indices = np.asarray(samples)

    # Validate inputs
    if not token_list:
        raise ValueError("Token list cannot be empty.")

    if expected.ndim != 1 or len(expected) != len(token_list):
        raise ValueError(
            "Token and probability counts must match."
        )

    if (
        not np.all(np.isfinite(expected))
        or np.any(expected < 0)
        or not np.isfinite(expected.sum())
        or expected.sum() <= 0
    ):
        raise ValueError(
            "Invalid probability distribution."
        )

    if sampled_indices.ndim != 1 or sampled_indices.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    if (
        sampled_indices.dtype.kind not in "iu"
        or np.any(sampled_indices < 0)
        or np.any(sampled_indices >= len(token_list))
    ):
        raise ValueError("Invalid sampled token indices.")

    # Normalize theoretical distribution
    expected = expected / expected.sum()

    # Calculate observed frequencies
    counts = np.bincount(
        sampled_indices,
        minlength=len(token_list),
    )

    observed = counts / sampled_indices.size

    # Prepare output path
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create grouped bar chart
    x = np.arange(len(token_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        ax.bar(
            x - width / 2,
            expected,
            width,
            label="Expected",
        )

        ax.bar(
            x + width / 2,
            observed,
            width,
            label="Observed",
        )

        ax.set_title(
            "Token Sampling: Expected vs Observed"
        )

        ax.set_xlabel("Tokens")
        ax.set_ylabel("Probability")

        ax.set_xticks(x)
        ax.set_xticklabels(token_list)

        ax.set_ylim(
            0,
            max(expected.max(), observed.max()) * 1.2,
        )

        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)

    return path
