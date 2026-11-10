"""Regression tests for Windows transient locks during JSON finalization."""

import json
from pathlib import Path

import pytest

from llm_sampling_lab.movie_reviews import Movie
from llm_sampling_lab.results import (
    ExperimentRecorder,
    MovieReviewRecorder,
    _write_json_atomically,
)


def _experiment_recorder(tmp_path):
    return ExperimentRecorder(
        model="test-model",
        prompt="Generate a slogan",
        experiment_type="temperature",
        parameter_values=[0.5],
        fixed_parameter="top_p",
        fixed_value=1.0,
        repetitions=1,
        output_dir=tmp_path,
    )


def _movie_recorder(tmp_path):
    return MovieReviewRecorder(
        model="test-model",
        movie=Movie(
            title="Test movie",
            genre="Drama",
            description="A brief fictional plot.",
        ),
        scenarios=((0.5, 1),),
        seed=42,
        output_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "recorder_factory",
    [_experiment_recorder, _movie_recorder],
    ids=["sampling", "movie_review"],
)
def test_recorder_retries_transient_replace_lock(
    tmp_path, monkeypatch, recorder_factory
):
    """Both recorders retry without losing the finished JSON report."""

    recorder = recorder_factory(tmp_path)
    original_replace = Path.replace
    attempts = []
    delays = []

    def flaky_replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError(13, "Temporary file lock")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(
        "llm_sampling_lab.results.time.sleep",
        delays.append,
    )

    recorder.complete()

    assert len(attempts) == 3
    assert delays == [0.05, 0.1]
    assert json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )["status"] == "completed"
    assert not recorder.file_path.with_suffix(".json.tmp").exists()


def test_failed_replace_preserves_old_report_and_temporary_data(
    tmp_path, monkeypatch
):
    """A persistent lock must not destroy the previously saved report."""

    destination = tmp_path / "report.json"
    destination.write_text('{"status": "running"}', encoding="utf-8")

    attempts = []
    delays = []

    def denied_replace(source, target):
        attempts.append((source, target))
        raise PermissionError(13, "File locked")

    monkeypatch.setattr(Path, "replace", denied_replace)
    monkeypatch.setattr(
        "llm_sampling_lab.results.time.sleep",
        delays.append,
    )

    with pytest.raises(PermissionError):
        _write_json_atomically(destination, {"status": "completed"})

    assert len(attempts) == 5  # 1 initial attempt + 4 retries
    assert delays == [0.05, 0.1, 0.2, 0.4]
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "running",
    }
    temporary = destination.with_suffix(".json.tmp")
    assert json.loads(temporary.read_text(encoding="utf-8")) == {
        "status": "completed",
    }


def test_non_permission_replace_errors_are_not_retried(
    tmp_path, monkeypatch
):
    destination = tmp_path / "report.json"
    attempts = []
    delays = []

    def failed_replace(source, target):
        attempts.append((source, target))
        raise OSError(5, "I/O error")

    monkeypatch.setattr(Path, "replace", failed_replace)
    monkeypatch.setattr(
        "llm_sampling_lab.results.time.sleep",
        delays.append,
    )

    with pytest.raises(OSError):
        _write_json_atomically(destination, {"status": "running"})

    assert len(attempts) == 1
    assert delays == []


def test_write_is_flushed_and_fsynced_before_replace(
    tmp_path, monkeypatch
):
    """Verify the temporary JSON exists and is complete before rename."""

    destination = tmp_path / "report.json"
    original_replace = Path.replace
    fsync_calls = []
    original_fsync = __import__("os").fsync

    def track_fsync(descriptor):
        fsync_calls.append(descriptor)
        original_fsync(descriptor)

    def verify_before_replace(source, target):
        assert fsync_calls
        assert json.loads(source.read_text(encoding="utf-8")) == {
            "text": "Türkçe veri",
        }
        return original_replace(source, target)

    monkeypatch.setattr("llm_sampling_lab.results.os.fsync", track_fsync)
    monkeypatch.setattr(Path, "replace", verify_before_replace)

    _write_json_atomically(destination, {"text": "Türkçe veri"})

    assert len(fsync_calls) == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "text": "Türkçe veri",
    }
