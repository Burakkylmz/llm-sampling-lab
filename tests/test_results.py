
import json

from llm_sampling_lab.experiments import SamplingResult
from llm_sampling_lab.results import ExperimentRecorder


def create_recorder(tmp_path):
    """Create an isolated test recorder."""

    return ExperimentRecorder(
        model="test-model",
        prompt="Generate a slogan.",
        experiment_type="temperature",
        parameter_values=[0.1, 0.5],
        fixed_parameter="top_p",
        fixed_value=1.0,
        repetitions=2,
        output_dir=tmp_path,
    )


def test_report_created(tmp_path):
    """A JSON report should exist when the experiment starts."""

    recorder = create_recorder(tmp_path)

    assert recorder.file_path.exists()

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert data["status"] == "running"
    assert data["model"] == "test-model"
    assert data["results"] == []


def test_incremental_result_persistence(tmp_path):
    """Successful responses should be saved immediately."""

    recorder = create_recorder(tmp_path)

    result = SamplingResult(
        experiment="temperature",
        temperature=0.1,
        top_p=1.0,
        repetition=1,
        response="Drive the Future.",
    )

    recorder.record(result)

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert len(data["results"]) == 1
    assert data["results"][0]["response"] == (
        "Drive the Future."
    )

    assert data["summary"][0]["unique_response_ratio"] == 1.0
    assert data["summary"][1]["unique_response_ratio"] is None


def test_completed_experiment(tmp_path):
    """A completed experiment should retain its results."""

    recorder = create_recorder(tmp_path)

    recorder.record(
        SamplingResult(
            experiment="temperature",
            temperature=0.1,
            top_p=1.0,
            repetition=1,
            response="Slogan A",
        )
    )

    recorder.record(
        SamplingResult(
            experiment="temperature",
            temperature=0.1,
            top_p=1.0,
            repetition=2,
            response="Slogan A",
        )
    )

    recorder.complete()

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert data["status"] == "completed"
    assert data["finished_at_utc"] is not None
    assert len(data["results"]) == 2
    assert data["summary"][0]["unique_response_ratio"] == 0.5


def test_failed_experiment_preserves_results(tmp_path):
    """A failed experiment must retain earlier responses."""

    recorder = create_recorder(tmp_path)

    recorder.record(
        SamplingResult(
            experiment="temperature",
            temperature=0.1,
            top_p=1.0,
            repetition=1,
            response="Saved response",
        )
    )

    recorder.fail(RuntimeError("Simulated API failure"))

    data = json.loads(
        recorder.file_path.read_text(encoding="utf-8")
    )

    assert data["status"] == "failed"
    assert data["error_type"] == "RuntimeError"
    assert len(data["results"]) == 1
    assert data["results"][0]["response"] == "Saved response"
