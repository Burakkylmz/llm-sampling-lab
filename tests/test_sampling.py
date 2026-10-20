
import numpy as np
import pytest

from llm_sampling_lab.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token_indices,
)


BASE_PROBABILITIES = [0.40, 0.25, 0.15, 0.12, 0.08]


def test_temperature_one_preserves_distribution():
    """T=1 should preserve the original distribution."""

    result = apply_temperature(BASE_PROBABILITIES, 1.0)

    np.testing.assert_allclose(
        result,
        BASE_PROBABILITIES,
        atol=1e-12,
    )


def test_low_temperature_sharpens_distribution():
    """Lower temperature increases the dominant probability."""

    result = apply_temperature(BASE_PROBABILITIES, 0.5)

    assert result[0] > BASE_PROBABILITIES[0]
    assert np.isclose(result.sum(), 1.0)


def test_high_temperature_flattens_distribution():
    """Higher temperature reduces the dominant probability."""

    result = apply_temperature(BASE_PROBABILITIES, 2.0)

    assert result[0] < BASE_PROBABILITIES[0]
    assert np.isclose(result.sum(), 1.0)


def test_zero_temperature_is_rejected():
    """Temperature zero is not supported by the simulator."""

    with pytest.raises(
        ValueError,
        match="Temperature must be",
    ):
        apply_temperature(BASE_PROBABILITIES, 0.0)



def test_top_k_one():
    """Top-k=1 should keep only the dominant token."""

    result = apply_top_k(BASE_PROBABILITIES, k=1)

    expected = [1.0, 0.0, 0.0, 0.0, 0.0]

    np.testing.assert_allclose(result, expected)


def test_top_k_three():
    """Top-k=3 should keep and renormalize three tokens."""

    result = apply_top_k(BASE_PROBABILITIES, k=3)

    expected = [0.5, 0.3125, 0.1875, 0.0, 0.0]

    np.testing.assert_allclose(result, expected)


def test_top_k_all_tokens():
    """Top-k equal to vocabulary size preserves distribution."""

    result = apply_top_k(BASE_PROBABILITIES, k=5)

    np.testing.assert_allclose(
        result,
        BASE_PROBABILITIES,
        atol=1e-12,
    )


def test_top_k_stable_ties():
    """Equal probabilities should preserve original token order."""

    probabilities = [0.4, 0.3, 0.3]

    result = apply_top_k(probabilities, k=2)

    np.testing.assert_allclose(
        result,
        [4 / 7, 3 / 7, 0.0],
    )


@pytest.mark.parametrize("invalid_k", [0, -1, 6, 1.5, True])
def test_invalid_top_k(invalid_k):
    """Invalid k values must be rejected."""

    with pytest.raises(ValueError, match="k must be"):
        apply_top_k(BASE_PROBABILITIES, invalid_k)


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.0, 0.0, 0.0],
        [0.5, -0.2, 0.7],
        [0.5, float("nan"), 0.5],
        [0.5, float("inf"), 0.5],
    ],
)
def test_invalid_top_k_probabilities(probabilities):
    """Invalid probability distributions must be rejected."""

    with pytest.raises(ValueError):
        apply_top_k(probabilities, k=2)



def test_top_p_low_threshold():
    """Low top-p should keep only the dominant token."""

    result = apply_top_p(BASE_PROBABILITIES, 0.3)

    np.testing.assert_allclose(
        result,
        [1.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_top_p_medium_threshold():
    """Top-p=0.7 should retain the first three tokens."""

    result = apply_top_p(BASE_PROBABILITIES, 0.7)

    expected = [0.5, 0.3125, 0.1875, 0.0, 0.0]

    np.testing.assert_allclose(result, expected)


def test_top_p_high_threshold():
    """Top-p=0.9 should retain four tokens."""

    result = apply_top_p(BASE_PROBABILITIES, 0.9)

    expected = [
        0.40 / 0.92,
        0.25 / 0.92,
        0.15 / 0.92,
        0.12 / 0.92,
        0.0,
    ]

    np.testing.assert_allclose(result, expected)


def test_top_p_one_preserves_distribution():
    """Top-p=1 should preserve the original distribution."""

    result = apply_top_p(BASE_PROBABILITIES, 1.0)

    np.testing.assert_allclose(
        result,
        BASE_PROBABILITIES,
        atol=1e-12,
    )


def test_top_p_stable_ties():
    """Equal probabilities should preserve original order."""

    probabilities = [0.4, 0.3, 0.3]

    result = apply_top_p(probabilities, 0.6)

    np.testing.assert_allclose(
        result,
        [4 / 7, 3 / 7, 0.0],
    )


def test_top_p_zero_probability_token():
    """Zero-probability tokens should stay at zero."""

    result = apply_top_p([0.6, 0.4, 0.0], 1.0)

    np.testing.assert_allclose(
        result,
        [0.6, 0.4, 0.0],
    )


@pytest.mark.parametrize(
    "invalid_top_p",
    [
        0.0,
        -0.1,
        1.1,
        True,
        float("nan"),
        float("inf"),
        "0.7",
    ],
)
def test_invalid_top_p(invalid_top_p):
    """Invalid nucleus thresholds must be rejected."""

    with pytest.raises(ValueError, match="top_p must be"):
        apply_top_p(BASE_PROBABILITIES, invalid_top_p)


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [0.0, 0.0],
        [0.5, -0.2, 0.7],
        [0.5, float("nan"), 0.5],
        [0.5, float("inf"), 0.5],
    ],
)
def test_invalid_top_p_probabilities(probabilities):
    """Invalid probability distributions must be rejected."""

    with pytest.raises(ValueError):
        apply_top_p(probabilities, 0.7)



def test_sampling_reproducibility():
    """The same seed should produce identical samples."""

    first = sample_token_indices(
        BASE_PROBABILITIES,
        n_samples=100,
        seed=42,
    )

    second = sample_token_indices(
        BASE_PROBABILITIES,
        n_samples=100,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_sampling_empirical_distribution():
    """Observed frequencies should approximate probabilities."""

    samples = sample_token_indices(
        BASE_PROBABILITIES,
        n_samples=10000,
        seed=42,
    )

    counts = np.bincount(samples, minlength=5)
    observed = counts / len(samples)

    np.testing.assert_allclose(
        observed,
        BASE_PROBABILITIES,
        atol=0.02,
    )


def test_sampling_after_top_k_one():
    """Top-k=1 should always select the dominant token."""

    filtered = apply_top_k(BASE_PROBABILITIES, k=1)

    samples = sample_token_indices(
        filtered,
        n_samples=100,
        seed=42,
    )

    assert np.all(samples == 0)


@pytest.mark.parametrize(
    "invalid_count",
    [0, -1, 1.5, True],
)
def test_invalid_sample_count(invalid_count):
    """Invalid sample counts must be rejected."""

    with pytest.raises(
        ValueError,
        match="n_samples must be",
    ):
        sample_token_indices(
            BASE_PROBABILITIES,
            n_samples=invalid_count,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [-1, 1.5, True],
)
def test_invalid_sampling_seed(invalid_seed):
    """Invalid random seeds must be rejected."""

    with pytest.raises(
        ValueError,
        match="seed must be",
    ):
        sample_token_indices(
            BASE_PROBABILITIES,
            n_samples=10,
            seed=invalid_seed,
        )
