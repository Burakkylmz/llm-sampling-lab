
from unittest.mock import MagicMock, patch

from llm_sampling_lab.experiments import (
    SamplingResult,
    run_temperature_experiment,
    run_top_p_experiment
)


@patch("llm_sampling_lab.experiments.generate_response")
def test_temperature_experiment_count(mock_generate):
    """Each temperature should run the configured repetitions."""

    mock_generate.return_value = "Test response"

    results = run_temperature_experiment(
        client=MagicMock(),
        model="test-model",
        prompt="Generate a slogan.",
        temperature_values=[0.1, 0.5, 1.0],
        top_p=1.0,
        repetitions=5,
    )

    assert len(results) == 15
    assert mock_generate.call_count == 15
    assert all(isinstance(r, SamplingResult) for r in results)


@patch("llm_sampling_lab.experiments.generate_response")
def test_temperature_experiment_parameters(mock_generate):
    """Only temperature should vary across the experiments."""

    mock_generate.return_value = "Test response"

    results = run_temperature_experiment(
        client=MagicMock(),
        model="test-model",
        prompt="Generate a slogan.",
        temperature_values=[0.1, 0.5, 1.0],
        top_p=1.0,
        repetitions=1,
    )

    assert [r.temperature for r in results] == [0.1, 0.5, 1.0]
    assert all(r.top_p == 1.0 for r in results)

    actual_temperatures = [
        call.kwargs["temperature"]
        for call in mock_generate.call_args_list
    ]

    assert actual_temperatures == [0.1, 0.5, 1.0]
    assert all(
        call.kwargs["top_p"] == 1.0
        for call in mock_generate.call_args_list
    )


@patch("llm_sampling_lab.experiments.generate_response")
def test_top_p_experiment_count(mock_generate):
    """Each top-p value should run the configured repetitions."""

    mock_generate.return_value = "Test response"

    results = run_top_p_experiment(
        client=MagicMock(),
        model="test-model",
        prompt="Generate a slogan.",
        top_p_values=[0.3, 0.7, 1.0],
        temperature=0.7,
        repetitions=5,
    )

    assert len(results) == 15
    assert mock_generate.call_count == 15

    assert all(
        isinstance(result, SamplingResult)
        for result in results
    )

    assert all(
        result.experiment == "top_p"
        for result in results
    )


@patch("llm_sampling_lab.experiments.generate_response")
def test_top_p_experiment_parameters(mock_generate):
    """Only top-p should vary between experiments."""

    mock_generate.return_value = "Test response"

    results = run_top_p_experiment(
        client=MagicMock(),
        model="test-model",
        prompt="Generate a slogan.",
        top_p_values=[0.3, 0.7, 1.0],
        temperature=0.7,
        repetitions=1,
    )

    assert [result.top_p for result in results] == [
        0.3, 0.7, 1.0
    ]

    assert all(
        result.temperature == 0.7
        for result in results
    )

    actual_top_p_values = [
        call.kwargs["top_p"]
        for call in mock_generate.call_args_list
    ]

    assert actual_top_p_values == [0.3, 0.7, 1.0]

    assert all(
        call.kwargs["temperature"] == 0.7
        for call in mock_generate.call_args_list
    )
