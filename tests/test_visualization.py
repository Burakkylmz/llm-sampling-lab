
import pytest

from llm_sampling_lab.visualization import (
    save_sampling_comparison,
)


def test_sampling_chart_created(tmp_path):
    """A valid sampling experiment should generate a PNG."""

    output_path = tmp_path / "sampling_chart.png"

    result = save_sampling_comparison(
        tokens=["AI", "Machine", "Deep"],
        probabilities=[0.5, 0.3, 0.2],
        samples=[0, 0, 1, 2, 0, 1],
        output_path=output_path,
    )

    assert result.exists()
    assert result.suffix == ".png"

    # Validate PNG file signature
    assert result.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_invalid_sample_indices(tmp_path):
    """Out-of-range token indices must be rejected."""

    with pytest.raises(
        ValueError,
        match="Invalid sampled token indices",
    ):
        save_sampling_comparison(
            tokens=["AI", "Machine"],
            probabilities=[0.6, 0.4],
            samples=[0, 1, 5],
            output_path=tmp_path / "invalid.png",
        )

