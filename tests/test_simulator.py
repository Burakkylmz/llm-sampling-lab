
from unittest.mock import MagicMock

import pytest
from pathlib import Path
from llm_sampling_lab import cli
from llm_sampling_lab.config import load_settings
from llm_sampling_lab.simulator import run_simulation_cli


@pytest.mark.parametrize(
    "strategy, expected_files",
    [
        ("temperature", 3),
        ("top-k", 3),
        ("top-p", 3),
    ],
)
def test_simulation_generates_visualizations(
    tmp_path,
    monkeypatch,
    strategy,
    expected_files,
):
    """Each strategy must generate PNG visualizations."""

    monkeypatch.chdir(tmp_path)

    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(project_root / "settings.toml")

    run_simulation_cli(
        settings=settings,
        strategy=strategy,
        samples=100,
        seed=42,
    )

    files = list(
        (tmp_path / "results/visualizations").glob(
            f"{strategy}_*.png"
        )
    )

    assert len(files) == expected_files

    for file in files:
        assert file.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
