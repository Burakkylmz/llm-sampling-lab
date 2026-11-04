
from random import Random


def build_execution_schedule(
    values: list[float | int],
    repetitions: int,
    *,
    randomize: bool = False,
    order_seed: int | None = None,
) -> list[tuple[float | int, int]]:
    """
    Build a reproducible sampling experiment schedule.

    Default:
        Preserve the existing parameter-grouped order.

    Randomized:
        Shuffle parameter values independently within
        each repetition round using a dedicated seed.
    """

    if not values:
        raise ValueError(
            "Experiment values cannot be empty."
        )

    if (
        type(repetitions) is not int
        or repetitions < 1
    ):
        raise ValueError(
            "repetitions must be a positive integer."
        )

    if type(randomize) is not bool:
        raise ValueError(
            "randomize must be a boolean."
        )

    if order_seed is not None and (
        type(order_seed) is not int
        or order_seed < 0
    ):
        raise ValueError(
            "order_seed must be a non-negative integer."
        )

    if randomize and order_seed is None:
        raise ValueError(
            "Randomized execution requires an order_seed."
        )

    if not randomize:
        if order_seed is not None:
            raise ValueError(
                "order_seed requires randomize=True."
            )

        return [
            (value, repetition)
            for value in values
            for repetition in range(1, repetitions + 1)
        ]

    rng = Random(order_seed)
    schedule = []

    for repetition in range(1, repetitions + 1):
        round_values = values.copy()
        rng.shuffle(round_values)

        schedule.extend(
            (value, repetition)
            for value in round_values
        )

    return schedule
