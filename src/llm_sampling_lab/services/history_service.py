
import json

from pathlib import Path
from typing import Any


SUPPORTED_EXPERIMENTS = {
    "temperature",
    "top_p",
    "top_k",
}


def list_experiment_reports(
    results_dir: str | Path = "results",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Load saved sampling experiments, newest first."""

    if limit < 1:
        raise ValueError("limit must be positive.")

    root = Path(results_dir)

    if not root.is_dir():
        return []

    reports = []

    for file_path in root.glob("*/*.json"):
        try:
            with file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                report = json.load(file)

        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        if not isinstance(report, dict):
            continue

        if report.get("experiment_type") not in SUPPORTED_EXPERIMENTS:
            continue

        if not isinstance(report.get("results"), list):
            continue

        reports.append({
            **report,
            "_file_path": str(file_path),
        })

    reports.sort(
        key=lambda report: str(
            report.get("started_at_utc") or ""
        ),
        reverse=True,
    )

    return reports[:limit]
