
import pytest

from llm_sampling_lab.services.execution_schedule import (
    build_execution_schedule,
)


def test_default_preserves_existing_order():
    schedule = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=2,
    )

    assert schedule == [
        (1, 1),
        (1, 2),
        (20, 1),
        (20, 2),
        (40, 1),
        (40, 2),
    ]


def test_randomized_schedule_is_reproducible():
    first = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=5,
        randomize=True,
        order_seed=42,
    )

    second = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=5,
        randomize=True,
        order_seed=42,
    )

    assert first == second
    assert len(first) == 15


def test_every_parameter_runs_once_per_round():
    schedule = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=3,
        randomize=True,
        order_seed=123,
    )

    for repetition in range(1, 4):
        values = [
            value
            for value, current_round in schedule
            if current_round == repetition
        ]

        assert sorted(values) == [1, 20, 40]


def test_all_original_experiments_are_preserved():
    grouped = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=4,
    )

    randomized = build_execution_schedule(
        values=[1, 20, 40],
        repetitions=4,
        randomize=True,
        order_seed=42,
    )

    assert sorted(grouped) == sorted(randomized)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": [], "repetitions": 3},
        {"values": [1], "repetitions": 0},
        {"values": [1], "repetitions": True},
        {
            "values": [1],
            "repetitions": 2,
            "randomize": True,
        },
        {
            "values": [1],
            "repetitions": 2,
            "randomize": True,
            "order_seed": -1,
        },
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_execution_schedule(**kwargs)
