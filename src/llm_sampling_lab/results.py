
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from collections.abc import Sequence

from llm_sampling_lab.movie_reviews import (
    Movie,
    MovieReviewResult,
)
from llm_sampling_lab.experiments import SamplingResult


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# A brief, bounded retry handles transient Windows locks from AV/indexers.
_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def _write_json_atomically(path: Path, data: dict) -> None:
    """Durably write a report, then replace the destination atomically.

    Never delete the existing report before replacement. If every retry
    fails, preserve the temporary file for recovery and raise the error.
    """

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())

    # The file is closed before trying to rename it.
    for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == len(_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS[attempt])


class ExperimentRecorder:
    """Persist sampling experiment results as JSON."""

    def __init__(
        self,
        model: str,
        prompt: str,
        experiment_type: str,
        parameter_values: list[float | int],
        fixed_parameter: str,
        fixed_value: float,
        repetitions: int,
        output_dir: str | Path = "results",
        provider: str = "groq",
        extra_fixed_parameters: dict[str, float] | None = None,
        warmup_requests: int = 0,
        seed: int | None = None,
        randomize_order: bool = False,
        order_seed: int | None = None,
    ) -> None:
        if experiment_type not in (
            "temperature",
            "top_p",
            "top_k",
        ):
            raise ValueError(
                f"Unsupported experiment type: {experiment_type}"
            )

        if repetitions < 1:
            raise ValueError(
                "repetitions must be a positive integer."
            )

        self.experiment_id = uuid4().hex
        self.experiment_type = experiment_type
        self.parameter_values = parameter_values.copy()
        self.results: list[SamplingResult] = []

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )

        directory = Path(output_dir) / experiment_type
        directory.mkdir(parents=True, exist_ok=True)

        self.file_path = directory / (
            f"{timestamp}_{self.experiment_id[:8]}.json"
        )

        self.report = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "experiment_type": experiment_type,
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "parameters": {
                "variable": experiment_type,
                "values": self.parameter_values,
                "fixed": {
                    fixed_parameter: fixed_value,
                    **(extra_fixed_parameters or {}),
                },
                "repetitions": repetitions,
                "warmup_requests": warmup_requests,
                "seed": seed,
                "randomize_order": randomize_order,
                "order_seed": order_seed,
            },
            "status": "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "error_type": None,
            "results": [],
            "summary": [],
        }

        self._save()

    def record(self, result: SamplingResult) -> None:
        """Save a successful API result immediately."""

        if self.report["status"] != "running":
            raise RuntimeError(
                "Cannot record results after experiment completion."
            )

        if result.experiment != self.experiment_type:
            raise ValueError("Experiment type mismatch.")

        self.results.append(result)

        self.report["results"] = [
            asdict(item) for item in self.results
        ]

        self.report["summary"] = self._build_summary()

        self._save()

    def complete(self) -> None:
        """Mark the experiment as completed."""

        self.report["status"] = "completed"
        self.report["finished_at_utc"] = _utc_now()

        self._save()

    def fail(self, error: Exception) -> None:
        """Mark the experiment as failed without losing results."""

        self.report["status"] = "failed"
        self.report["finished_at_utc"] = _utc_now()

        # Store the exception type, not potentially sensitive details.
        self.report["error_type"] = type(error).__name__

        self._save()

    def _build_summary(self) -> list[dict]:
        """Calculate diversity metrics for each parameter value."""

        summary = []

        for value in self.parameter_values:

            group = [
                result
                for result in self.results
                if (
                    (
                        self.experiment_type == "temperature"
                        and result.temperature == value
                    )
                    or (
                        self.experiment_type == "top_p"
                        and result.top_p == value
                    )
                    or (
                        self.experiment_type == "top_k"
                        and result.top_k == value
                    )
                )
            ]

            responses = [result.response for result in group]

            total = len(responses)
            unique_count = len(set(responses))

            summary.append({
                "parameter_value": value,
                "total_responses": total,
                "unique_responses": unique_count,
                "unique_response_ratio": (
                    unique_count / total if total else None
                ),
            })

        return summary

    def _save(self) -> None:
        """Atomically write the JSON report."""
        _write_json_atomically(self.file_path, self.report)



class MovieReviewRecorder:
    """Persist movie-review experiment results incrementally."""

    def __init__(
        self,
        model: str,
        movie: Movie,
        scenarios: Sequence[tuple[float, int]],
        seed: int,
        output_dir: str | Path = "results",
    ) -> None:

        self.results: list[MovieReviewResult] = []

        experiment_id = uuid4().hex

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )

        directory = Path(output_dir) / "movie_reviews"
        directory.mkdir(parents=True, exist_ok=True)

        self.file_path = directory / (
            f"{timestamp}_{experiment_id[:8]}.json"
        )

        scenario_config = [
            {
                "temperature": temperature,
                "local_top_k": top_k,
            }
            for temperature, top_k in scenarios
        ]

        self.report = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "experiment_type": "movie_reviews",
            "model": model,
            "movie": asdict(movie),
            "parameters": {
                "seed": seed,
                "groq_top_p": 1.0,
                "scenarios": scenario_config,
                "top_k_scope": "local_review_style_selection",
            },
            "status": "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "error_type": None,
            "results": [],
            "summary": {
                "total_scenarios": len(scenario_config),
                "completed_scenarios": 0,
                "validation_warning_count": 0,
            },
        }

        self._save()

    def record(self, result: MovieReviewResult) -> None:
        """Persist a completed scenario immediately."""

        if self.report["status"] != "running":
            raise RuntimeError(
                "Cannot record results after experiment completion."
            )

        expected_scenario = len(self.results) + 1

        if result.scenario != expected_scenario:
            raise ValueError(
                "Movie review scenario sequence mismatch."
            )

        self.results.append(result)

        self.report["results"] = [
            asdict(item) for item in self.results
        ]

        self.report["summary"]["completed_scenarios"] = (
            len(self.results)
        )

        self.report["summary"]["validation_warning_count"] = sum(
            bool(item.validation_issues)
            for item in self.results
        )

        self._save()

    def complete(self) -> None:
        """Mark a successful experiment as completed."""

        if self.report["status"] != "running":
            raise RuntimeError("Experiment is no longer running.")

        self.report["status"] = "completed"
        self.report["finished_at_utc"] = _utc_now()

        self._save()

    def fail(self, error: Exception) -> None:
        """Preserve results and record the failure type."""

        self.report["status"] = "failed"
        self.report["finished_at_utc"] = _utc_now()
        self.report["error_type"] = type(error).__name__

        self._save()

    def _save(self) -> None:
        """Write the JSON report using atomic replacement."""
        _write_json_atomically(self.file_path, self.report)
