
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path: Path, *arguments: str):
    """Execute the real CLI entry point in an isolated directory."""

    # Copy configuration into the isolated test directory
    source = PROJECT_ROOT / "settings.toml"
    destination = tmp_path / "settings.toml"

    if not destination.exists():
        shutil.copyfile(source, destination)

    # Deliberately remove the API key
    environment = os.environ.copy()
    environment.pop("GROQ_API_KEY", None)

    command = [
        sys.executable,
        "-c",
        (
            "from llm_sampling_lab.cli import main; "
            "raise SystemExit(main())"
        ),
        *arguments,
    ]

    return subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "temperature",
        "top-k",
        "top-p",
    ],
)
def test_offline_cli_creates_visualizations(
    tmp_path,
    strategy,
):
    """Each simulation strategy should work without an API key."""

    result = run_cli(
        tmp_path,
        "--simulate",
        strategy,
        "--samples",
        "100",
        "--seed",
        "42",
    )

    assert result.returncode == 0, result.stderr

    assert f"Simulation: {strategy}" in result.stdout

    output_dir = tmp_path / "results" / "visualizations"

    files = list(output_dir.glob(f"{strategy}_*.png"))

    assert len(files) == 3

    for file in files:
        assert file.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--simulate", "top-k", "--samples", "0"),
        ("--simulate", "top-p", "--seed", "-1"),
        ("--samples", "100"),
        ("--seed", "42"),
        (
            "--simulate",
            "temperature",
            "--experiment",
            "top-p",
        ),
    ],
)
def test_invalid_cli_arguments_are_rejected(
    tmp_path,
    arguments,
):
    """Invalid CLI combinations must fail."""

    result = run_cli(tmp_path, *arguments)

    assert result.returncode != 0


def test_cli_sampling_is_reproducible(tmp_path):
    """The same strategy and seed should reproduce the output."""

    arguments = (
        "--simulate",
        "top-k",
        "--samples",
        "1000",
        "--seed",
        "42",
    )

    first = run_cli(tmp_path, *arguments)
    second = run_cli(tmp_path, *arguments)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert first.stdout == second.stdout
