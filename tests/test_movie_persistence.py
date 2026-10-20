
import json
import sys
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab import cli, movie_reviews
from llm_sampling_lab.movie_reviews import DEFAULT_MOVIE


VALID_REVIEW = (
    DEFAULT_MOVIE.title + " " + "hikaye " * 80
)


@pytest.fixture
def mock_movie_client(monkeypatch):
    """Avoid real Groq API requests."""

    monkeypatch.setattr(
        cli,
        "create_groq_client",
        lambda: MagicMock(),
    )


def test_movie_reviews_json_persistence(
    monkeypatch,
    tmp_path,
    mock_movie_client,
):
    """Four successful scenarios must be saved to JSON."""

    monkeypatch.chdir(tmp_path)

    mock_generate = MagicMock(
        return_value=VALID_REVIEW
    )

    monkeypatch.setattr(
        movie_reviews,
        "generate_response",
        mock_generate,
    )

    cli.run_movie_reviews_cli(
        settings={"groq": {"model": "test-model"}},
    )

    reports = list(
        (tmp_path / "results" / "movie_reviews").glob(
            "*.json"
        )
    )

    assert len(reports) == 1

    data = json.loads(
        reports[0].read_text(encoding="utf-8")
    )

    assert mock_generate.call_count == 4
    assert data["status"] == "completed"
    assert data["experiment_type"] == "movie_reviews"
    assert len(data["results"]) == 4
    assert data["summary"]["completed_scenarios"] == 4
    assert data["summary"]["validation_warning_count"] == 0
    assert data["parameters"]["top_k_scope"] == (
        "local_review_style_selection"
    )


def test_movie_reviews_partial_failure(
    monkeypatch,
    tmp_path,
    mock_movie_client,
):
    """An API failure must preserve earlier successful results."""

    monkeypatch.chdir(tmp_path)

    mock_generate = MagicMock(
        side_effect=[
            VALID_REVIEW,
            RuntimeError("Simulated API failure"),
        ]
    )

    monkeypatch.setattr(
        movie_reviews,
        "generate_response",
        mock_generate,
    )

    with pytest.raises(RuntimeError):
        cli.run_movie_reviews_cli(
            settings={"groq": {"model": "test-model"}},
        )

    reports = list(
        (tmp_path / "results" / "movie_reviews").glob(
            "*.json"
        )
    )

    assert len(reports) == 1

    data = json.loads(
        reports[0].read_text(encoding="utf-8")
    )

    assert data["status"] == "failed"
    assert data["error_type"] == "RuntimeError"
    assert len(data["results"]) == 1
    assert data["summary"]["completed_scenarios"] == 1


def test_movie_reviews_cli_routing(monkeypatch, capsys):
    """The CLI must route movie-reviews to its runner."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["sampling-lab", "--experiment", "movie-reviews"],
    )

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: {
            "groq": {"model": "test-model"},
            "experiments": {"repetitions": 5},
        },
    )

    monkeypatch.setattr(
        cli,
        "validate_settings",
        lambda settings: None,
    )

    mock_runner = MagicMock()

    monkeypatch.setattr(
        cli,
        "run_movie_reviews_cli",
        mock_runner,
    )

    cli.main()

    mock_runner.assert_called_once_with(
        settings={
            "groq": {"model": "test-model"},
            "experiments": {"repetitions": 5},
        },
    )

    captured = capsys.readouterr()
    assert "Repetitions:" not in captured.out


def test_movie_reviews_repetitions_rejected(monkeypatch):
    """Movie reviews do not accept --repetitions."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-lab",
            "--experiment",
            "movie-reviews",
            "--repetitions",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
