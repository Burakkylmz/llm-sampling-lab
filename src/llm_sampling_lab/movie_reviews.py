
from dataclasses import dataclass
from typing import Sequence
from collections.abc import Callable
from groq import Groq

from llm_sampling_lab.groq_client import generate_response
from llm_sampling_lab.sampling import (
    apply_temperature,
    apply_top_k,
    sample_token_indices,
)


REVIEW_STYLES = (
    "Positive",
    "Critical",
    "Technical",
    "Emotional",
)

STYLE_PROBABILITIES = (0.4, 0.3, 0.2, 0.1)

DEFAULT_SCENARIOS = (
    (0.3, 1),
    (0.3, 3),
    (1.0, 1),
    (1.0, 3),
)


@dataclass(frozen=True)
class Movie:
    title: str
    genre: str
    description: str


@dataclass(frozen=True)
class MovieReviewResult:
    scenario: int
    temperature: float
    local_top_k: int
    review_style: str
    review: str
    validation_issues: tuple[str, ...]


DEFAULT_MOVIE = Movie(
    title="Son İstasyon",
    genre="Science Fiction / Drama",
    description=(
        "A scientist discovers that humanity has only "
        "one year to establish a colony on another planet. "
        "She must choose between her family and the mission."
    ),
)


def select_review_style(
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    """Select a review style using local temperature and top-k."""

    distribution = apply_temperature(
        STYLE_PROBABILITIES,
        temperature,
    )

    filtered = apply_top_k(
        distribution,
        k=top_k,
    )

    selected = sample_token_indices(
        probabilities=filtered,
        n_samples=1,
        seed=seed,
    )

    return REVIEW_STYLES[int(selected[0])]


def build_review_prompt(
    movie: Movie,
    review_style: str,
) -> str:
    """Build a constrained movie-review generation prompt."""

    if review_style not in REVIEW_STYLES:
        raise ValueError("Unsupported review style.")

    if not all((
        movie.title.strip(),
        movie.genre.strip(),
        movie.description.strip(),
    )):
        raise ValueError("Movie information cannot be empty.")

    return f"""
You are a professional movie critic.

Write a Turkish review for a fictional movie.

MOVIE INFORMATION:
Title: {movie.title}
Genre: {movie.genre}
Plot: {movie.description}

REVIEW STYLE:
{review_style}

STRICT RULES:
1. Begin the review with the exact movie title.
2. Do not change the movie title.
3. Do not invent character names.
4. Do not invent actors, directors, or awards.
5. Do not invent details about acting, music,
   cinematography, or visual effects.
6. Discuss only the provided plot, its themes,
   and potential emotional impact.
7. Follow the selected review style.
8. Write between 80 and 100 Turkish words.
9. Return only the review.
""".strip()


def validate_review(
    review: str,
    movie: Movie,
) -> tuple[str, ...]:
    """Perform deterministic structural checks on a review."""

    issues = []

    if movie.title.casefold() not in review.casefold():
        issues.append("movie_title_missing_or_changed")

    # Simple whitespace-based word count.
    word_count = len(review.split())

    if not 80 <= word_count <= 100:
        issues.append("word_count_out_of_range")

    return tuple(issues)



def run_movie_review_experiment(
    client: Groq,
    model: str,
    movie: Movie = DEFAULT_MOVIE,
    scenarios: Sequence[tuple[float, int]] = DEFAULT_SCENARIOS,
    seed: int = 42,
    on_result: Callable[[MovieReviewResult], None] | None = None,
) -> list[MovieReviewResult]:
    """Run movie-review scenarios using local sampling and Groq."""

    results: list[MovieReviewResult] = []

    for index, (temperature, top_k) in enumerate(
        scenarios,
        start=1,
    ):
        style = select_review_style(
            temperature=temperature,
            top_k=top_k,
            seed=seed + index - 1,
        )

        prompt = build_review_prompt(
            movie=movie,
            review_style=style,
        )

        review = generate_response(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=1.0,
        )

        issues = validate_review(
            review=review,
            movie=movie,
        )

        result = MovieReviewResult(
            scenario=index,
            temperature=temperature,
            local_top_k=top_k,
            review_style=style,
            review=review,
            validation_issues=issues,
        )

        results.append(result)

        # Save each successful response immediately.
        if on_result is not None:
            on_result(result)

    return results
