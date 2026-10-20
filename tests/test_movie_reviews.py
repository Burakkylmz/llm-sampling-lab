
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import movie_reviews
from llm_sampling_lab.movie_reviews import (
    DEFAULT_MOVIE,
    Movie,
    REVIEW_STYLES,
    build_review_prompt,
    run_movie_review_experiment,
    select_review_style,
    validate_review,
)


def test_top_k_one_selects_dominant_style():
    """Top-k=1 must always select the dominant review style."""

    for seed in (1, 42, 100):
        result = select_review_style(
            temperature=0.3,
            top_k=1,
            seed=seed,
        )

        assert result == "Positive"


def test_style_selection_is_reproducible():
    """The same seed should produce the same style."""

    first = select_review_style(1.0, 3, seed=42)
    second = select_review_style(1.0, 3, seed=42)

    assert first == second
    assert first in REVIEW_STYLES


def test_review_prompt_contains_movie_information():
    """The prompt must contain the exact movie information."""

    prompt = build_review_prompt(
        DEFAULT_MOVIE,
        "Critical",
    )

    assert DEFAULT_MOVIE.title in prompt
    assert DEFAULT_MOVIE.description in prompt
    assert "Critical" in prompt
    assert "Do not invent character names" in prompt


def test_unsupported_review_style():
    """An unknown review style should be rejected."""

    with pytest.raises(ValueError):
        build_review_prompt(
            DEFAULT_MOVIE,
            "Unknown",
        )


def test_review_validation_detects_missing_title():
    """A changed movie title should be flagged."""

    review = "This is a fictional review."

    issues = validate_review(
        review,
        DEFAULT_MOVIE,
    )

    assert "movie_title_missing_or_changed" in issues
    assert "word_count_out_of_range" in issues


def test_valid_review_structure():
    """A review with the exact title and valid length passes."""

    review = (
        DEFAULT_MOVIE.title
        + " "
        + "etkileyici " * 80
    )

    issues = validate_review(
        review,
        DEFAULT_MOVIE,
    )

    assert issues == ()


def test_movie_review_experiment_uses_groq(
    monkeypatch,
):
    """The runner should invoke Groq with the expected parameters."""

    mock_generate = MagicMock(
        return_value="Son İstasyon etkileyici bir film."
    )

    monkeypatch.setattr(
        movie_reviews,
        "generate_response",
        mock_generate,
    )

    client = MagicMock()

    results = run_movie_review_experiment(
        client=client,
        model="test-model",
        scenarios=[
            (0.3, 1),
            (1.0, 3),
        ],
        seed=42,
    )

    assert len(results) == 2
    assert mock_generate.call_count == 2

    assert results[0].temperature == 0.3
    assert results[0].local_top_k == 1
    assert results[0].review_style == "Positive"

    assert results[1].temperature == 1.0
    assert results[1].local_top_k == 3

    for call in mock_generate.call_args_list:
        assert call.kwargs["top_p"] == 1.0
        assert call.kwargs["model"] == "test-model"

    assert results[0].validation_issues == (
        "word_count_out_of_range",
    )


def test_invalid_local_top_k():
    """Invalid local top-k values must be rejected."""

    with pytest.raises(ValueError):
        select_review_style(
            temperature=0.3,
            top_k=5,
            seed=42,
        )
