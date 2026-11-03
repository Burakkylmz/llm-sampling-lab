
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    results_dir = Path("results/top_k")

    reports = sorted(
        results_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )

    if not reports:
        raise SystemExit("No Top-k JSON reports found.")

    report_path = reports[-1]

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    if report["status"] != "completed":
        raise SystemExit(
            "The latest experiment is not completed."
        )

    summary = report["summary"]

    top_k_values = [
        str(item["parameter_value"])
        for item in summary
    ]

    diversity = [
        item["unique_response_ratio"] * 100
        for item in summary
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(top_k_values, diversity)

    ax.set_title("Top-k Sampling: Response Diversity")
    ax.set_xlabel("Top-k Value")
    ax.set_ylabel("Unique Response Ratio (%)")
    ax.set_ylim(0, 100)

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=4,
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    output_path = report_path.with_name(
        f"{report_path.stem}_diversity.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Report: {report_path}")
    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
