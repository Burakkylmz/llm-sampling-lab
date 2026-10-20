
import math
from collections.abc import Sequence

import numpy as np


def apply_temperature(
    probabilities: Sequence[float],
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to token probabilities."""

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "Temperature must be a positive finite number."
        )

    probs = np.asarray(probabilities, dtype=np.float64)

    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(
            "Probabilities must be a non-empty 1D array."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "Probabilities must contain only finite values."
        )

    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError(
            "Probabilities must be non-negative "
            "with a positive total."
        )

    # Normalize initial probabilities
    probs = probs / probs.sum()

    # Preserve zero-probability tokens
    positive = probs > 0

    # Stable temperature scaling in log space
    logits = np.log(probs[positive]) / temperature
    logits = logits - np.max(logits)

    weights = np.exp(logits)

    scaled = np.zeros_like(probs)
    scaled[positive] = weights / weights.sum()

    return scaled



def apply_top_k(
    probabilities: Sequence[float],
    k: int,
) -> np.ndarray:
    """Keep the k highest-probability tokens and renormalize."""

    probs = np.asarray(probabilities, dtype=np.float64)

    # Validate probability distribution
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(
            "Probabilities must be a non-empty 1D array."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "Probabilities must contain only finite values."
        )

    if np.any(probs < 0):
        raise ValueError(
            "Probabilities must be non-negative."
        )

    total = probs.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Probabilities must have a positive finite total."
        )

    # Validate k
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or not 1 <= k <= probs.size
    ):
        raise ValueError(
            f"k must be an integer between 1 and {probs.size}."
        )

    # Normalize original probabilities
    probs = probs / total

    # Sort by descending probability
    # Stable ordering preserves original order for ties
    sorted_indices = np.argsort(-probs, kind="stable")

    # Keep only the highest-probability k tokens
    selected_indices = sorted_indices[:k]

    filtered = np.zeros_like(probs)
    filtered[selected_indices] = probs[selected_indices]

    # Renormalize remaining probabilities
    filtered = filtered / filtered.sum()

    return filtered



def apply_top_p(
    probabilities: Sequence[float],
    top_p: float,
) -> np.ndarray:
    """Apply nucleus sampling filtering and renormalization."""

    # Validate top-p threshold
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError(
            "top_p must be a finite number between 0 (exclusive) and 1."
        )

    probs = np.asarray(probabilities, dtype=np.float64)

    # Validate probability distribution
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(
            "Probabilities must be a non-empty 1D array."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "Probabilities must contain only finite values."
        )

    if np.any(probs < 0):
        raise ValueError(
            "Probabilities must be non-negative."
        )

    total = probs.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Probabilities must have a positive finite total."
        )

    # Normalize input probabilities
    probs = probs / total

    # Sort tokens by descending probability
    sorted_indices = np.argsort(-probs, kind="stable")
    sorted_probs = probs[sorted_indices]

    # Calculate cumulative probability
    cumulative_probs = np.cumsum(sorted_probs)

    # Protect against floating-point accumulation error
    cumulative_probs[-1] = 1.0

    # Find the smallest prefix reaching the threshold
    cutoff_index = np.searchsorted(
        cumulative_probs,
        top_p,
        side="left",
    )

    selected_indices = sorted_indices[:cutoff_index + 1]

    # Remove tokens outside the nucleus
    filtered = np.zeros_like(probs)
    filtered[selected_indices] = probs[selected_indices]

    # Renormalize selected probabilities
    filtered = filtered / filtered.sum()

    return filtered



def sample_token_indices(
    probabilities: Sequence[float],
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample token indices from a probability distribution."""

    # Validate sample count
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, int)
        or n_samples < 1
    ):
        raise ValueError(
            "n_samples must be a positive integer."
        )

    # Validate random seed
    if seed is not None:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer or None."
            )

    probs = np.asarray(probabilities, dtype=np.float64)

    # Validate probability distribution
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(
            "Probabilities must be a non-empty 1D array."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError(
            "Probabilities must contain only finite values."
        )

    if np.any(probs < 0):
        raise ValueError(
            "Probabilities must be non-negative."
        )

    total = probs.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "Probabilities must have a positive finite total."
        )

    probs = probs / total

    # Create reproducible random generator
    rng = np.random.default_rng(seed)

    # Sample token indices according to probabilities
    selected_indices = rng.choice(
        a=probs.size,
        size=n_samples,
        replace=True,
        p=probs,
    )

    return selected_indices
