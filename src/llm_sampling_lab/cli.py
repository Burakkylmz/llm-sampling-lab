
import argparse
import sys
import tomllib
from llm_sampling_lab.results import ExperimentRecorder
from typing import Any, Literal
from llm_sampling_lab.simulator import run_simulation_cli

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from llm_sampling_lab.config import (
    load_settings,
    validate_settings,
)

from llm_sampling_lab.groq_client import (
    create_groq_client,
    generate_response,
)

from llm_sampling_lab.experiments import (
    SamplingResult,
    run_temperature_experiment,
    run_top_p_experiment,
)


SAMPLING_PROMPT = (
    "Create one short and memorable advertising slogan "
    "for an electric car brand. Return only the slogan."
)


def check_api(model: str) -> None:
    """Verify Groq API connectivity."""

    with create_groq_client() as client:
        response = generate_response(
            client=client,
            model=model,
            prompt="Reply with exactly: Connection successful",
        )

    print(f"Groq API response: {response}")


def _display_experiment_results(
    results: list[SamplingResult],
    parameter: Literal["temperature", "top_p"],
    values: list[float],
) -> None:
    """Display grouped sampling results and diversity metrics."""

    for value in values:

        group = [
            result
            for result in results
            if (
                result.temperature == value
                if parameter == "temperature"
                else result.top_p == value
            )
        ]

        responses = [result.response for result in group]

        total = len(responses)
        unique_count = len(set(responses))
        unique_ratio = unique_count / total if total else 0.0

        print("\n" + "=" * 60)

        if parameter == "temperature":
            print(f"TEMPERATURE: {value}")

            if group:
                print(f"TOP-P: {group[0].top_p}")

        else:
            print(f"TOP-P: {value}")

            if group:
                print(f"TEMPERATURE: {group[0].temperature}")

        print("=" * 60)

        for result in group:
            print(f"{result.repetition}. {result.response}")

        print("-" * 60)
        print(f"Total Responses: {total}")
        print(f"Unique Responses: {unique_count}")
        print(f"Unique Response Ratio: {unique_ratio:.2f}")



def run_temperature_cli(
    settings: dict[str, Any],
    repetitions: int,
) -> None:
    """Run temperature experiments with persistent JSON reporting."""

    experiment_settings = settings["temperature"]

    recorder = ExperimentRecorder(
        model=settings["groq"]["model"],
        prompt=SAMPLING_PROMPT,
        experiment_type="temperature",
        parameter_values=experiment_settings["values"],
        fixed_parameter="top_p",
        fixed_value=experiment_settings["top_p"],
        repetitions=repetitions,
    )

    try:
        with create_groq_client() as client:
            results = run_temperature_experiment(
                client=client,
                model=settings["groq"]["model"],
                prompt=SAMPLING_PROMPT,
                temperature_values=experiment_settings["values"],
                top_p=experiment_settings["top_p"],
                repetitions=repetitions,
                on_result=recorder.record,
            )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    _display_experiment_results(
        results=results,
        parameter="temperature",
        values=experiment_settings["values"],
    )

    print(f"\nJSON Report: {recorder.file_path}")


def run_top_p_cli(
    settings: dict[str, Any],
    repetitions: int,
) -> None:
    """Run top-p experiments with persistent JSON reporting."""

    experiment_settings = settings["top_p"]

    recorder = ExperimentRecorder(
        model=settings["groq"]["model"],
        prompt=SAMPLING_PROMPT,
        experiment_type="top_p",
        parameter_values=experiment_settings["values"],
        fixed_parameter="temperature",
        fixed_value=experiment_settings["temperature"],
        repetitions=repetitions,
    )

    try:
        with create_groq_client() as client:
            results = run_top_p_experiment(
                client=client,
                model=settings["groq"]["model"],
                prompt=SAMPLING_PROMPT,
                top_p_values=experiment_settings["values"],
                temperature=experiment_settings["temperature"],
                repetitions=repetitions,
                on_result=recorder.record,
            )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    _display_experiment_results(
        results=results,
        parameter="top_p",
        values=experiment_settings["values"],
    )

    print(f"\nJSON Report: {recorder.file_path}")



def main() -> None:
    """Entry point for the LLM Sampling Lab."""

    # CLI argument parsing
    parser = argparse.ArgumentParser(
        description="LLM Sampling Laboratory"
    )

    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument(
        "--check-api",
        action="store_true",
        help="Test Groq API connectivity.",
    )

    action_group.add_argument(
        "--experiment",
        choices=["temperature", "top-p"],
        help="Select a sampling experiment.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override configured experiment repetitions.",
    )

    action_group.add_argument(
        "--simulate",
        choices=["temperature", "top-k", "top-p"],
        help="Run an offline sampling simulation.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of token samples (default: 10000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: 42).",
    )


    args = parser.parse_args()


    if not args.simulate and (
        args.samples is not None
        or args.seed is not None
    ):
        parser.error(
            "--samples and --seed require --simulate."
        )

    if args.samples is not None and args.samples < 1:
        parser.error(
            "--samples must be a positive integer."
        )

    if args.seed is not None and args.seed < 0:
        parser.error(
            "--seed must be non-negative."
        )


    # CLI argument validation
    if args.repetitions is not None:

        if not args.experiment:
            parser.error(
                "--repetitions requires --experiment."
            )

        if args.repetitions < 1:
            parser.error(
                "--repetitions must be a positive integer."
            )

    print("LLM Sampling Lab")
    print("----------------")

    try:
        # Load and validate configuration
        settings = load_settings()
        validate_settings(settings)

        # Resolve effective repetitions
        effective_repetitions = (
            args.repetitions
            if args.repetitions is not None
            else settings["experiments"]["repetitions"]
        )

        print("Configuration loaded successfully.")
        print(f"Model: {settings['groq']['model']}")

        if args.experiment:
            print(f"Repetitions: {effective_repetitions}")

        # Command routing
        if args.check_api:
            check_api(settings["groq"]["model"])

        elif args.experiment == "temperature":
            run_temperature_cli(
                settings=settings,
                repetitions=effective_repetitions,
            )

        elif args.experiment == "top-p":
            run_top_p_cli(
                settings=settings,
                repetitions=effective_repetitions,
            )

        elif args.simulate:
            run_simulation_cli(
                settings=settings,
                strategy=args.simulate,
                samples=(
                    args.samples
                    if args.samples is not None
                    else 10000
                ),
                seed=(
                    args.seed
                    if args.seed is not None
                    else 42
                ),
            )


    # Groq API error handling
    except AuthenticationError:
        print(
            "API error: Invalid Groq API key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except RateLimitError:
        print(
            "API error: Groq rate limit exceeded.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except APIConnectionError:
        print(
            "API error: Unable to connect to Groq.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except APIStatusError as exc:
        print(
            f"API error: HTTP {exc.status_code}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    # Configuration and response error handling
    except (
        FileNotFoundError,
        ValueError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(
            f"Configuration or response error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    print("\nOperation completed successfully.")


if __name__ == "__main__":
    main()
