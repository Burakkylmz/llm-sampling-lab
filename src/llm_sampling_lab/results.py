
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from llm_sampling_lab.experiments import SamplingResult


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class ExperimentRecorder:
    """Persist sampling experiment results as JSON."""

    def __init__(
        self,
        model: str,
        prompt: str,
        experiment_type: str,
        parameter_values: list[float],
        fixed_parameter: str,
        fixed_value: float,
        repetitions: int,
        output_dir: str | Path = "results",
    ) -> None:

        if experiment_type not in ("temperature", "top_p"):
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
            "model": model,
            "prompt": prompt,
            "parameters": {
                "variable": experiment_type,
                "values": self.parameter_values,
                "fixed": {
                    fixed_parameter: fixed_value,
                },
                "repetitions": repetitions,
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
                    result.temperature == value
                    if self.experiment_type == "temperature"
                    else result.top_p == value
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

        temporary_path = self.file_path.with_suffix(
            ".json.tmp"
        )

        content = json.dumps(
            self.report,
            indent=2,
            ensure_ascii=False,
        )

        temporary_path.write_text(
            content,
            encoding="utf-8",
        )

        temporary_path.replace(self.file_path)
