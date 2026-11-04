
import argparse
import os
import sys
import tomllib

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Literal

from groq import (
    APIConnectionError as GroqConnectionError,
    APIStatusError as GroqStatusError,
    AuthenticationError as GroqAuthenticationError,
    RateLimitError as GroqRateLimitError,
)

from openai import (
    OpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIStatusError as OpenAIStatusError,
    AuthenticationError as OpenAIAuthenticationError,
    RateLimitError as OpenAIRateLimitError,
)

from llm_sampling_lab.config import (
    load_settings,
    validate_settings,
)

from llm_sampling_lab.domain.models import (
    GenerationRequest,
    SamplingResult,
)

from llm_sampling_lab.groq_client import (
    create_groq_client,
    generate_response,
)

from llm_sampling_lab.movie_reviews import (
    DEFAULT_MOVIE,
    DEFAULT_SCENARIOS,
    run_movie_review_experiment,
)

from llm_sampling_lab.providers.base import BaseProvider

from llm_sampling_lab.providers.factory import (
    build_provider_registry,
    create_provider,
)

from llm_sampling_lab.results import (
    ExperimentRecorder,
    MovieReviewRecorder,
)

from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)

from llm_sampling_lab.simulator import run_simulation_cli

from llm_sampling_lab.providers.model_catalog import (
    get_model_metadata,
    list_catalog_models,
)

from llm_sampling_lab.providers.model_discovery import (
    SupportsModelDiscovery,
)

from llm_sampling_lab.providers.model_catalog import (
    list_catalog_models,
)

from ollama import Client as OllamaClient


SAMPLING_PROMPT = (
    "Create one short and memorable advertising slogan "
    "for an electric car brand. Return only the slogan."
)
TOP_K_VALUES = [1, 20, 40]
TOP_K_TEMPERATURE = 0.8
TOP_K_TOP_P = 0.9


@contextmanager
def provider_session(
    name: str,
) -> Iterator[BaseProvider]:
    """Create and manage a provider for one operation."""

    if name == "groq":
        with create_groq_client() as client:
            yield create_provider(
                "groq",
                groq_client=client,
            )

    elif name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing or empty."
            )

        with OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        ) as client:
            yield create_provider(
                "openai",
                openai_client=client,
            )

    elif name == "ollama":
        with OllamaClient(
            host="http://127.0.0.1:11434",
            timeout=120.0,
        ) as client:
            yield create_provider(
                "ollama",
                ollama_client=client,
            )

    else:
        raise ValueError(
            f"Unsupported provider: {name}"
        )



def resolve_model(
    settings: dict[str, Any],
    provider_name: str,
    model_override: str | None,
) -> str:
    """Resolve the selected model."""

    if model_override is not None:
        model = model_override.strip()

        if not model:
            raise ValueError("Model cannot be empty.")

        return model

    if provider_name == "groq":
        return settings["groq"]["model"]

    raise ValueError(
        f"--model is required for provider '{provider_name}'."
    )


def check_api(model: str) -> None:
    """Preserve the existing Groq API check."""

    with create_groq_client() as client:
        response = generate_response(
            client=client,
            model=model,
            prompt="Reply with exactly: Connection successful",
        )

    print(f"Groq API response: {response}")


def check_provider_api(
    provider_name: str,
    model: str,
) -> None:
    """Check connectivity through the provider abstraction."""

    with provider_session(provider_name) as provider:
        response = provider.generate(
            GenerationRequest(
                model=model,
                prompt="Reply with exactly: Connection successful",
            )
        )

    print(
        f"{provider_name} API response: {response.text}"
    )


def _display_experiment_results(
    results: list[SamplingResult],
    parameter: Literal["temperature", "top_p"],
    values: list[float],
) -> None:
    """Display experiment responses and diversity metrics."""

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

        responses = [
            result.response for result in group
        ]

        total = len(responses)
        unique_count = len(set(responses))
        unique_ratio = (
            unique_count / total if total else 0.0
        )

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
            print(
                f"{result.repetition}. {result.response}"
            )

        print("-" * 60)
        print(f"Total Responses: {total}")
        print(f"Unique Responses: {unique_count}")
        print(
            f"Unique Response Ratio: {unique_ratio:.2f}"
        )


def run_temperature_cli(
    settings: dict[str, Any],
    repetitions: int,
    provider_name: str = "groq",
    model: str | None = None,
) -> None:
    """Run temperature experiments with a selected provider."""

    experiment_settings = settings["temperature"]

    selected_model = resolve_model(
        settings,
        provider_name,
        model,
    )

    recorder = ExperimentRecorder(
        model=selected_model,
        provider=provider_name,
        prompt=SAMPLING_PROMPT,
        experiment_type="temperature",
        parameter_values=experiment_settings["values"],
        fixed_parameter="top_p",
        fixed_value=experiment_settings["top_p"],
        repetitions=repetitions,
    )

    try:
        with provider_session(provider_name) as provider:
            service = ExperimentService(
                provider=provider,
                model=selected_model,
            )

            results = service.run_temperature(
                prompt=SAMPLING_PROMPT,
                temperatures=experiment_settings["values"],
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
    provider_name: str = "groq",
    model: str | None = None,
) -> None:
    """Run top-p experiments with a selected provider."""

    experiment_settings = settings["top_p"]

    selected_model = resolve_model(
        settings,
        provider_name,
        model,
    )

    recorder = ExperimentRecorder(
        model=selected_model,
        provider=provider_name,
        prompt=SAMPLING_PROMPT,
        experiment_type="top_p",
        parameter_values=experiment_settings["values"],
        fixed_parameter="temperature",
        fixed_value=experiment_settings["temperature"],
        repetitions=repetitions,
    )

    try:
        with provider_session(provider_name) as provider:
            service = ExperimentService(
                provider=provider,
                model=selected_model,
            )

            results = service.run_top_p(
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


def run_movie_reviews_cli(
    settings: dict[str, Any],
) -> None:
    """Run the existing Groq movie-review experiment."""

    model = settings["groq"]["model"]

    recorder = MovieReviewRecorder(
        model=model,
        movie=DEFAULT_MOVIE,
        scenarios=DEFAULT_SCENARIOS,
        seed=42,
    )

    try:
        with create_groq_client() as client:
            results = run_movie_review_experiment(
                client=client,
                model=model,
                movie=DEFAULT_MOVIE,
                scenarios=DEFAULT_SCENARIOS,
                seed=42,
                on_result=recorder.record,
            )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    for result in results:
        print("\n" + "=" * 60)
        print(f"SCENARIO: {result.scenario}")
        print(f"Temperature: {result.temperature}")
        print(f"Local Top-k: {result.local_top_k}")
        print(f"Review Style: {result.review_style}")
        print("=" * 60)

        print(result.review)

        if result.validation_issues:
            print(
                "\nValidation Warnings: "
                + ", ".join(result.validation_issues)
            )
        else:
            print(
                "\nValidation: Passed structural checks"
            )

    print(f"\nJSON Report: {recorder.file_path}")


def show_capabilities(
    provider_name: str,
    model: str,
) -> None:
    """Display model capabilities without calling an API."""

    provider = create_provider(provider_name)
    capabilities = provider.get_capabilities(model)

    print("\nModel Capabilities")
    print("=" * 50)
    print(f"Provider: {capabilities.provider}")
    print(f"Model: {capabilities.model}")
    
    metadata = get_model_metadata(
        provider_name,
        model,
    )

    print(f"Source: {metadata.source}")
    print(
        "Last verified: "
        f"{metadata.last_verified or 'not recorded'}"
    )
    print(
        "Live API verified: "
        f"{metadata.live_verified}"
    )

    print("-" * 50)

    for name in ("temperature", "top_p", "top_k"):
        status = getattr(capabilities, name)
        description = status.value

        if status.value == "supported":
            limits = capabilities.parameter_ranges.get(name)

            if limits is not None:
                left = (
                    "[" if limits.minimum_inclusive else "("
                )
                right = (
                    "]" if limits.maximum_inclusive else ")"
                )
                description += (
                    f" | Range: {left}{limits.minimum}, "
                    f"{limits.maximum}{right}"
                )

        print(f"{name}: {description}")


def show_model_list(provider_name: str) -> None:
    """List models registered in the local catalog."""

    models = list_catalog_models(provider_name)

    print(f"\nAvailable catalog models: {provider_name}")
    print("=" * 50)

    if not models:
        print("No models registered.")
        return

    for model in models:
        print(f"- {model}")

    print("\nSource: Local static model catalog.")
    print("This is not a live API model listing.")



def discover_models_cli(provider_name: str) -> None:
    """Display models returned by the provider API."""

    with provider_session(provider_name) as provider:
        if not isinstance(provider, SupportsModelDiscovery):
            raise ValueError(
                f"{provider_name} does not support model discovery."
            )

        models = provider.discover_models()

    catalog_models = set(
        list_catalog_models(provider_name)
    )

    print(f"\nLive models: {provider_name}")
    print("=" * 60)

    if not models:
        print("The provider returned no models.")
        return

    for item in models:
        catalog_status = (
            "known"
            if item.model_id in catalog_models
            else "not in local catalog"
        )

        print(f"\nModel: {item.model_id}")
        print(f"Owner: {item.owned_by or 'unknown'}")
        print(f"Catalog: {catalog_status}")

        if item.active is not None:
            print(f"Active: {item.active}")

    print(f"\nTotal models: {len(models)}")
    print("Source: Live provider API")
    print(
        "Note: Model listing does not verify "
        "sampling parameter support."
    )



def _display_top_k_results(
    results: list[SamplingResult],
    values: list[int],
) -> None:
    """Display Top-k response diversity."""

    for top_k in values:
        group = [
            result
            for result in results
            if result.top_k == top_k
        ]

        responses = [
            result.response
            for result in group
        ]

        total = len(responses)
        unique_count = len(set(responses))

        ratio = unique_count / total if total else 0.0

        print("\n" + "=" * 60)
        print(f"TOP-K: {top_k}")
        print(f"Total Responses: {total}")
        print(f"Unique Responses: {unique_count}")
        print(f"Unique Response Ratio: {ratio:.2f}")
        print("=" * 60)

        for result in group:
            print(
                f"{result.repetition}. {result.response}"
            )


def run_top_k_cli(
    settings: dict[str, Any],
    repetitions: int,
    provider_name: str = "ollama",
    model: str | None = None,
) -> None:
    """Run native Top-k experiments with Ollama."""

    if provider_name != "ollama":
        raise ValueError(
            "Native Top-k experiments currently require Ollama."
        )

    selected_model = resolve_model(
        settings,
        provider_name,
        model,
    )

    recorder = ExperimentRecorder(
        model=selected_model,
        provider=provider_name,
        prompt=SAMPLING_PROMPT,
        experiment_type="top_k",
        parameter_values=TOP_K_VALUES,
        fixed_parameter="temperature",
        fixed_value=TOP_K_TEMPERATURE,
        extra_fixed_parameters={
            "top_p": TOP_K_TOP_P,
        },
        repetitions=repetitions,
    )

    try:
        with provider_session(provider_name) as provider:
            service = ExperimentService(
                provider=provider,
                model=selected_model,
            )

            results = service.run_top_k(
                prompt=SAMPLING_PROMPT,
                top_k_values=TOP_K_VALUES,
                temperature=TOP_K_TEMPERATURE,
                top_p=TOP_K_TOP_P,
                repetitions=repetitions,
                on_result=recorder.record,
            )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    _display_top_k_results(
        results=results,
        values=TOP_K_VALUES,
    )

    print(f"\nJSON Report: {recorder.file_path}")


def main() -> None:
    """LLM Sampling Lab command-line entry point."""

    parser = argparse.ArgumentParser(
        description="LLM Sampling Laboratory"
    )

    parser.add_argument(
        "--provider",
        choices=build_provider_registry().available(),
        default="groq",
        help="LLM provider (default: groq).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model identifier.",
    )

    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument(
        "--check-api",
        action="store_true",
        help="Check provider API connectivity.",
    )

    action_group.add_argument(
        "--experiment",
        choices=[
            "temperature",
            "top-p",
            "top-k",
            "movie-reviews",
        ],
        help="Select a sampling experiment.",
    )

    action_group.add_argument(
        "--simulate",
        choices=[
            "temperature",
            "top-k",
            "top-p",
        ],
        help="Run an offline simulation.",
    )

    action_group.add_argument(
        "--show-capabilities",
        action="store_true",
        help="Show model sampling capabilities.",
    )

    action_group.add_argument(
        "--list-models",
        action="store_true",
        help="List models in the local catalog.",
    )

    action_group.add_argument(
        "--discover-models",
        action="store_true",
        help="List models using the live provider API.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override experiment repetitions.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Offline sample count (default: 10000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Offline random seed (default: 42).",
    )

    args = parser.parse_args()

    if (
        args.experiment == "top-k"
        and args.provider != "ollama"
    ):
        parser.error(
            "Native Top-k experiments currently require "
            "--provider ollama."
        )

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

    if args.repetitions is not None:
        if args.experiment not in (
            "temperature",
            "top-p",
            "top-k",
        ):
            parser.error(
                "--repetitions requires a sampling experiment."
            )

        if args.repetitions < 1:
            parser.error(
                "--repetitions must be a positive integer."
            )

    if args.simulate and (
        args.provider != "groq"
        or args.model is not None
    ):
        parser.error(
            "Offline simulations do not use "
            "--provider or --model."
        )

    if (
        args.experiment == "movie-reviews"
        and args.provider != "groq"
    ):
        parser.error(
            "Movie reviews currently support Groq only."
        )

    if (
        args.provider in ("openai", "ollama")
        and args.model is None
        and not args.simulate
        and not args.list_models
        and not args.discover_models
    ):
        parser.error(
            f"--provider {args.provider} requires --model."
        )

    if args.discover_models and args.model is not None:
        parser.error(
            "--discover-models does not accept --model."
        )

    if args.list_models and args.model is not None:
        parser.error(
            "--list-models does not accept --model."
        )

    if args.experiment == "movie-reviews":
        if args.provider != "groq":
            parser.error(
                "Movie reviews currently support Groq only."
            )

        if args.model is not None:
            parser.error(
                "Movie reviews currently use the configured "
                "Groq model; --model is not supported."
            )
            
    print("LLM Sampling Lab")
    print("----------------")

    try:
        settings = load_settings()
        validate_settings(settings)

        if args.list_models:
            show_model_list(args.provider)
            return

        if args.discover_models:
            discover_models_cli(args.provider)
            return

        selected_model = resolve_model(
            settings,
            args.provider,
            args.model,
        )

        repetitions = (
            args.repetitions
            if args.repetitions is not None
            else settings["experiments"]["repetitions"]
        )

        print("Configuration loaded successfully.")
        print(f"Provider: {args.provider}")
        print(f"Model: {selected_model}")

        if args.experiment in ("temperature", "top-p"):
            print(f"Repetitions: {repetitions}")

        if args.show_capabilities:
            show_capabilities(
                provider_name=args.provider,
                model=selected_model,
            )
        elif args.check_api:
            if args.provider == "groq":
                check_api(selected_model)
            else:
                check_provider_api(
                    args.provider,
                    selected_model,
                )

        elif args.experiment == "temperature":
            run_temperature_cli(
                settings=settings,
                repetitions=repetitions,
                provider_name=args.provider,
                model=selected_model,
            )

        elif args.experiment == "top-p":
            run_top_p_cli(
                settings=settings,
                repetitions=repetitions,
                provider_name=args.provider,
                model=selected_model,
            )

        elif args.experiment == "top-k":
            run_top_k_cli(
                settings=settings,
                repetitions=repetitions,
                provider_name=args.provider,
                model=selected_model,
            )

        elif args.experiment == "movie-reviews":
            run_movie_reviews_cli(
                settings=settings,
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

    except (
        GroqAuthenticationError,
        OpenAIAuthenticationError,
    ):
        print(
            "API error: Invalid API key.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except (
        GroqRateLimitError,
        OpenAIRateLimitError,
    ):
        print(
            "API error: Rate limit exceeded.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except (
        GroqConnectionError,
        OpenAIConnectionError,
    ):
        print(
            "API error: Connection failed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    except (
        GroqStatusError,
        OpenAIStatusError,
    ) as exc:
        print(
            f"API error: HTTP {exc.status_code}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

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