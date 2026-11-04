
import json

from llm_sampling_lab.services.history_service import (
    list_experiment_reports,
)


def write_report(
    root,
    experiment,
    name,
    started_at,
):
    folder = root / experiment
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / name

    path.write_text(
        json.dumps({
            "experiment_type": experiment,
            "provider": "ollama",
            "model": "llama3.2:1b",
            "started_at_utc": started_at,
            "status": "completed",
            "results": [
                {"response": "Hello"}
            ],
            "summary": [],
        }),
        encoding="utf-8",
    )


def test_history_returns_newest_first(tmp_path):
    write_report(
        tmp_path,
        "top_k",
        "old.json",
        "2026-11-01T10:00:00+00:00",
    )

    write_report(
        tmp_path,
        "top_k",
        "new.json",
        "2026-11-04T10:00:00+00:00",
    )

    reports = list_experiment_reports(tmp_path)

    assert len(reports) == 2
    assert "new.json" in reports[0]["_file_path"]


def test_history_skips_invalid_json(tmp_path):
    folder = tmp_path / "top_k"
    folder.mkdir()

    (folder / "broken.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert list_experiment_reports(tmp_path) == []


def test_history_ignores_other_experiments(tmp_path):
    write_report(
        tmp_path,
        "movie_reviews",
        "review.json",
        "2026-11-04T10:00:00+00:00",
    )

    assert list_experiment_reports(tmp_path) == []
