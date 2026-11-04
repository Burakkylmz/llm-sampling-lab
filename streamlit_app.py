
import pandas as pd
import streamlit as st
import json

from llm_sampling_lab.cli import provider_session
from llm_sampling_lab.config import (
    load_settings,
    validate_settings,
)
from llm_sampling_lab.domain.models import SamplingResult
from llm_sampling_lab.providers.factory import (
    build_provider_registry,
)
from llm_sampling_lab.services.experiment_service import (
    ExperimentService,
)
from pathlib import Path

from llm_sampling_lab.results import ExperimentRecorder

from llm_sampling_lab.services.history_service import (
    list_experiment_reports,
)
from llm_sampling_lab.services.comparison_service import (
    compare_experiment_reports,
    compare_multiple_experiment_reports,
)
from llm_sampling_lab.services.analytics_service import (
    summarize_sampling_results,
    build_comparison_insights,
)
from llm_sampling_lab.services.performance_service import (
    build_performance_rows,
    summarize_performance_rows,
)
from llm_sampling_lab.services.reproducibility_service import (
    summarize_reproducibility,
)
from llm_sampling_lab.services.benchmark_reporting_service import (
    build_token_comparison_rows,
    compare_benchmark_reports,
)
from llm_sampling_lab.services.controlled_benchmark import (
    ControlledBenchmarkConfig,
    run_controlled_benchmark,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM Sampling Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DEFAULTS = {
    "groq": "openai/gpt-oss-20b",
    "openai": "gpt-4.1-mini",
    "ollama": "llama3.2:1b",
}

EXPERIMENT_OPTIONS = [
    "temperature",
    "top-p",
    "top-k",
]

TOP_K_VALUES = [1, 20, 40]
TOP_K_TEMPERATURE = 0.8
TOP_K_TOP_P = 0.9

DEFAULT_PROMPT = (
    "Create one short and memorable advertising slogan "
    "for an electric car brand. Return only the slogan."
)


# ============================================================
# CONFIGURATION
# ============================================================

@st.cache_data
def get_settings() -> dict:
    """Load and validate experiment settings."""

    settings = load_settings()
    validate_settings(settings)

    return settings


def get_available_providers() -> list[str]:
    """Return registered LLM providers."""

    registry = build_provider_registry()
    return registry.available()


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================


def execute_experiment(
    provider_name: str,
    model: str,
    strategy: str,
    prompt: str,
    repetitions: int,
    settings: dict,
    warmup_requests: int = 0,
    seed: int | None = None,
    randomize_order: bool = False,
    order_seed: int | None = None,
) -> tuple[list[SamplingResult], str]:
    """Execute and persist an LLM sampling experiment."""

    if not model.strip():
        raise ValueError("Model cannot be empty.")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if repetitions < 1:
        raise ValueError("Invalid repetitions.")

    if strategy == "top-k" and provider_name != "ollama":
        raise ValueError(
            "Native Top-k experiments require Ollama."
        )

    if strategy == "temperature":
        config = settings["temperature"]

        experiment_type = "temperature"
        values = config["values"]
        fixed_parameter = "top_p"
        fixed_value = config["top_p"]
        extra_fixed = None

    elif strategy == "top-p":
        config = settings["top_p"]

        experiment_type = "top_p"
        values = config["values"]
        fixed_parameter = "temperature"
        fixed_value = config["temperature"]
        extra_fixed = None

    elif strategy == "top-k":
        experiment_type = "top_k"
        values = TOP_K_VALUES
        fixed_parameter = "temperature"
        fixed_value = TOP_K_TEMPERATURE
        extra_fixed = {"top_p": TOP_K_TOP_P}

    else:
        raise ValueError(
            f"Unsupported experiment: {strategy}"
        )

    recorder = ExperimentRecorder(
        model=model,
        provider=provider_name,
        prompt=prompt,
        experiment_type=experiment_type,
        parameter_values=values,
        fixed_parameter=fixed_parameter,
        fixed_value=fixed_value,
        extra_fixed_parameters=extra_fixed,
        repetitions=repetitions,
        warmup_requests=warmup_requests,
        seed=seed,
        randomize_order=randomize_order,
        order_seed=order_seed,
    )

    try:
        with provider_session(provider_name) as provider:
            service = ExperimentService(
                provider=provider,
                model=model,
                warmup_requests=warmup_requests,
                seed=seed,
                randomize_order=randomize_order,
                order_seed=order_seed,
            )

            if strategy == "temperature":
                results = service.run_temperature(
                    prompt=prompt,
                    temperatures=values,
                    top_p=fixed_value,
                    repetitions=repetitions,
                    on_result=recorder.record,
                )

            elif strategy == "top-p":
                results = service.run_top_p(
                    prompt=prompt,
                    top_p_values=values,
                    temperature=fixed_value,
                    repetitions=repetitions,
                    on_result=recorder.record,
                )

            else:
                results = service.run_top_k(
                    prompt=prompt,
                    top_k_values=values,
                    temperature=TOP_K_TEMPERATURE,
                    top_p=TOP_K_TOP_P,
                    repetitions=repetitions,
                    on_result=recorder.record,
                )

        recorder.complete()

    except Exception as exc:
        recorder.fail(exc)
        raise

    return results, str(recorder.file_path)


# ============================================================
# DATA PROCESSING
# ============================================================

def results_to_dataframe(
    results: list[SamplingResult],
) -> pd.DataFrame:
    """Convert experiment results into a DataFrame."""

    rows = []

    for item in results:

        rows.append(
            {
                "Experiment": item.experiment,
                "Temperature": item.temperature,
                "Top-p": item.top_p,
                "Top-k": item.top_k,
                "Repetition": item.repetition,
                "Response": item.response,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(
    results: list[SamplingResult],
    strategy: str,
) -> pd.DataFrame:
    """Calculate response diversity by sampling parameter."""

    parameter = {
        "temperature": "temperature",
        "top-p": "top_p",
        "top-k": "top_k",
    }[strategy]

    grouped = {}

    for item in results:

        value = getattr(item, parameter)

        grouped.setdefault(value, []).append(
            item.response
        )

    rows = []

    for value, responses in sorted(grouped.items()):

        total = len(responses)
        unique_count = len(set(responses))

        unique_ratio = (
            unique_count / total
            if total > 0
            else 0.0
        )

        rows.append(
            {
                "Parameter": value,
                "Total Responses": total,
                "Unique Responses": unique_count,
                "Unique Ratio": round(
                    unique_ratio,
                    3,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
    settings: dict,
) -> dict:
    """Render experiment configuration controls."""

    providers = get_available_providers()

    with st.sidebar:

        st.title("Experiment Configuration")

        st.divider()

        provider = st.selectbox(
            "LLM Provider",
            options=providers,
            index=0,
        )

        model = st.text_input(
            "Model",
            value=MODEL_DEFAULTS.get(
                provider,
                "",
            ),
            key=f"model_{provider}",
        )

        experiment = st.selectbox(
            "Sampling Strategy",
            options=EXPERIMENT_OPTIONS,
        )

        repetitions = st.number_input(
            "Repetitions",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
        )

        prompt = st.text_area(
            "Prompt",
            value=DEFAULT_PROMPT,
            height=150,
        )

        st.divider()

        # ----------------------------------------------------
        # EXPERIMENT PARAMETER INFORMATION
        # ----------------------------------------------------

        st.subheader("Sampling Parameters")

        if experiment == "temperature":

            config = settings["temperature"]

            st.write("**Temperature values**")

            st.code(
                str(config["values"]),
                language=None,
            )

            st.write(
                f"Fixed Top-p: {config['top_p']}"
            )

        elif experiment == "top-p":

            config = settings["top_p"]

            st.write("**Top-p values**")

            st.code(
                str(config["values"]),
                language=None,
            )

            st.write(
                "Fixed Temperature: "
                f"{config['temperature']}"
            )

        else:

            st.write("**Top-k values**")

            st.code(
                str(TOP_K_VALUES),
                language=None,
            )

            st.write(
                f"Fixed Temperature: {TOP_K_TEMPERATURE}"
            )

            st.write(
                f"Fixed Top-p: {TOP_K_TOP_P}"
            )

            if provider != "ollama":

                st.warning(
                    "Native Top-k experiments currently "
                    "require Ollama."
                )

        st.divider()
        
        st.subheader("Reproducibility Settings")

        warmup_requests = st.selectbox(
            "Warm-up requests",
            options=[0, 1, 2, 3],
            index=1 if provider == "ollama" else 0,
            help=(
                "Warm-up responses are excluded from "
                "recorded benchmark results. "
                "They still consume compute and, for "
                "cloud providers, may incur API charges."
            ),
        )

        use_fixed_seed = st.checkbox(
            "Enable fixed seed (Ollama only)",
            value=False,
            disabled=provider != "ollama",
        )

        seed = None

        if use_fixed_seed and provider == "ollama":
            seed = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    value=42,
                    step=1,
                )
            )

            st.warning(
                "Fixed seed is intended for reproducibility "
                "tests. Reusing it across repetitions may "
                "reduce observed response diversity."
            )

        
        st.divider()
        st.subheader("Execution Order")

        randomize_order = st.checkbox(
            "Randomize experiment order",
            value=False,
            help=(
                "Shuffle sampling parameter values "
                "within each repetition round."
            ),
        )

        order_seed = None

        if randomize_order:
            order_seed = int(
                st.number_input(
                    "Execution order seed",
                    min_value=0,
                    value=42,
                    step=1,
                    help=(
                        "Controls the order of parameter "
                        "execution, not LLM generation."
                    ),
                )
            )

            st.info(
                "Each repetition round will test "
                "all parameter values in a shuffled order."
            )

        run_button = st.button(
            "Run Experiment",
            type="primary",
            use_container_width=True,
            disabled=(
                experiment == "top-k"
                and provider != "ollama"
            ),
        )

        st.caption(
            "Experiments use the existing "
            "provider architecture and ExperimentService."
        )

    return {
        "provider": provider,
        "model": model,
        "experiment": experiment,
        "repetitions": int(repetitions),
        "prompt": prompt,
        "run_button": run_button,
        "warmup_requests": int(warmup_requests),
        "seed": seed,
        "randomize_order": randomize_order,
        "order_seed": order_seed,
    }


# ============================================================
# DASHBOARD HEADER
# ============================================================


def render_header(config: dict) -> None:
    """Render a clean professional dashboard header."""

    with st.container(border=True):
        st.caption("EXPERIMENTATION WORKSPACE")
        st.title("LLM Sampling Lab")

        st.write(
            "Explore and evaluate Temperature, Top-p "
            "and Top-k sampling across LLM providers."
        )

        st.caption(
            f"Active provider: {config['provider']}  |  "
            f"Model: {config['model']}  |  "
            f"Strategy: {config['experiment']}"
        )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Providers",
            len(get_available_providers()),
            border=True,
        )

    with col2:
        st.metric(
            "Strategies",
            len(EXPERIMENT_OPTIONS),
            border=True,
        )

    with col3:
        st.metric(
            "Repetitions",
            config["repetitions"],
            border=True,
        )

    with col4:
        last_run = st.session_state.get("last_run")

        st.metric(
            "Last Run Responses",
            len(last_run["results"]) if last_run else 0,
            border=True,
        )


# ============================================================
# EXPERIMENT TAB
# ============================================================

def render_experiment_tab(
    config: dict,
) -> None:
    """Display selected experiment configuration."""

    st.subheader("Experiment Setup")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Provider:** {config['provider']}"
        )

        st.write(
            f"**Model:** {config['model']}"
        )

    with col2:

        st.write(
            f"**Strategy:** {config['experiment']}"
        )

        st.write(
            f"**Repetitions:** {config['repetitions']}"
        )

    st.divider()

    st.markdown("**Prompt**")

    st.write(config["prompt"])

    st.info(
        "Configure your experiment in the sidebar "
        "and click Run Experiment."
    )

    last_run = st.session_state.get("last_run")

    if last_run:

        st.success(
            "Last experiment completed successfully."
        )

        st.write(
            "**Generated responses:** "
            f"{len(last_run['results'])}"
        )


# ============================================================
# RESULTS TAB
# ============================================================

def render_results_tab() -> None:
    """Display generated model responses."""

    st.subheader("Experiment Results")

    last_run = st.session_state.get("last_run")

    if last_run is None:

        st.info(
            "No experiment has been run yet."
        )

        return

    results = last_run["results"]

    st.caption(
        f"Provider: {last_run['provider']} | "
        f"Model: {last_run['model']} | "
        f"Strategy: {last_run['strategy']}"
    )

    st.divider()

    results_df = results_to_dataframe(results)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Responses",
            len(results),
        )

    with col2:

        unique_count = len(
            set(item.response for item in results)
        )

        st.metric(
            "Unique Responses",
            unique_count,
        )

    with col3:

        ratio = (
            unique_count / len(results)
            if results
            else 0.0
        )

        st.metric(
            "Unique Ratio",
            f"{ratio:.1%}",
        )

    st.divider()

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Results CSV",
        data=results_df.to_csv(
            index=False,
        ).encode("utf-8-sig"),
        file_name="sampling_results.csv",
        mime="text/csv",
    )

    report_path = Path(last_run["report_path"])

    st.caption(f"JSON Report: {report_path}")

    if report_path.is_file():
        st.download_button(
            label="Download JSON Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/json",
        )


# ============================================================
# ANALYTICS TAB
# ============================================================


def render_analytics_tab() -> None:
    """Render the advanced analytics dashboard."""

    st.subheader("Advanced Analytics")
    st.caption(
        "Analyze sampling diversity, response counts "
        "and differences between parameter groups."
    )

    last_run = st.session_state.get("last_run")

    if last_run is None:
        st.info("Run an experiment to view analytics.")
        return

    results = last_run["results"]
    strategy = last_run["strategy"]

    summary_rows = summarize_sampling_results(
        results,
        strategy,
    )

    if not summary_rows:
        st.info("No results available.")
        return

    summary = pd.DataFrame(summary_rows)

    total_responses = len(results)
    unique_responses = len({
        result.response for result in results
    })

    group_count = len(summary)

    diversity_range = (
        summary["Diversity (%)"].max()
        - summary["Diversity (%)"].min()
    )

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Responses",
            total_responses,
            border=True,
        )

    with col2:
        st.metric(
            "Unique Responses",
            unique_responses,
            border=True,
        )

    with col3:
        st.metric(
            "Parameter Groups",
            group_count,
            border=True,
        )

    with col4:
        st.metric(
            "Diversity Range",
            f"{diversity_range:.1f} pp",
            border=True,
        )

    st.divider()

    st.markdown("### Sampling Diversity Summary")

    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
    )

    # Make parameter values categorical for the charts.
    chart_df = summary.copy()
    chart_df["Parameter"] = (
        chart_df["Parameter"].astype(str)
    )
    chart_df = chart_df.set_index("Parameter")

    st.markdown("### Unique Response Ratio")

    st.bar_chart(
        chart_df[["Diversity (%)"]],
        sort=False,
        width="stretch",
    )

    st.markdown("### Response Counts")

    st.bar_chart(
        chart_df[
            ["Total Responses", "Unique Responses"]
        ],
        stack=False,
        sort=False,
        width="stretch",
    )

    st.markdown("### Diversity Difference")

    st.caption(
        "Percentage-point change relative to the "
        "lowest tested parameter value."
    )

    st.bar_chart(
        chart_df[["Delta from baseline (pp)"]],
        sort=False,
        width="stretch",
    )

    st.divider()

    st.info(
        "Unique-response ratios are descriptive measures. "
        "They depend on the number of repetitions, prompt, "
        "model and sampling settings. Higher diversity "
        "does not necessarily mean higher response quality."
    )

    st.divider()
    
    st.markdown("### Reproducibility Analysis")

    reproducibility = summarize_reproducibility(
        results,
        strategy,
    )

    st.dataframe(
        pd.DataFrame(reproducibility),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Exact Match indicates that all recorded responses "
        "for a parameter group are identical. "
        "For controlled reproducibility experiments, "
        "use a fixed seed with the same model and settings."
    )


# ============================================================
# EXPERIMENT HANDLER
# ============================================================

def handle_experiment_run(
    config: dict,
    settings: dict,
) -> None:
    """Run an experiment when requested."""

    if not config["run_button"]:
        return

    # Clear previous results before a new experiment.
    st.session_state.pop(
        "last_run",
        None,
    )

    try:

        with st.spinner(
            "Running LLM experiment..."
        ):

            results, report_path = execute_experiment(
                provider_name=config["provider"],
                model=config["model"].strip(),
                strategy=config["experiment"],
                prompt=config["prompt"].strip(),
                repetitions=config["repetitions"],
                settings=settings,
                warmup_requests=config["warmup_requests"],
                seed=config["seed"],
                randomize_order=config["randomize_order"],
                order_seed=config["order_seed"],
            )

        st.session_state["last_run"] = {
            "provider": config["provider"],
            "model": config["model"].strip(),
            "strategy": config["experiment"],
            "prompt": config["prompt"].strip(),
            "results": results,
            "report_path": report_path,
            "warmup_requests": config["warmup_requests"],
            "seed": config["seed"],
            "randomize_order": config["randomize_order"],
            "order_seed": config["order_seed"],
        }

        st.success(
            "Experiment completed successfully. "
            f"Generated {len(results)} responses."
        )

    except Exception as exc:

        st.error(
            "Experiment failed: "
            f"{type(exc).__name__}: {exc}"
        )



def render_history_tab() -> None:
    """Display previously saved experiment reports."""

    st.subheader("Experiment History")

    reports = list_experiment_reports()

    if not reports:
        st.info(
            "No saved experiments found. "
            "Run an experiment to create a report."
        )
        return

    st.caption(
        f"{len(reports)} saved experiment(s) found."
    )

    selected_index = st.selectbox(
        "Select an experiment",
        options=list(range(len(reports))),
        format_func=lambda index: (
            f"{reports[index].get('started_at_utc', 'Unknown date')} "
            f"| {reports[index].get('provider', 'unknown')} "
            f"| {reports[index].get('model', 'unknown')} "
            f"| {reports[index].get('experiment_type', 'unknown')}"
        ),
    )

    report = reports[selected_index]

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Provider",
            report.get("provider", "unknown"),
        )

    with col2:
        st.metric(
            "Experiment",
            report.get("experiment_type", "unknown"),
        )

    with col3:
        st.metric(
            "Responses",
            len(report["results"]),
        )

    st.write(
        f"**Model:** {report.get('model', 'unknown')}"
    )

    st.write(
        f"**Status:** {report.get('status', 'unknown')}"
    )

    st.write(
        f"**Started:** "
        f"{report.get('started_at_utc', 'unknown')}"
    )

    st.markdown("### Sampling Parameters")

    parameters = report.get("parameters", {})

    if isinstance(parameters, dict):
        st.json(parameters)

    st.markdown("### Saved Responses")

    if report["results"]:
        st.dataframe(
            pd.DataFrame(report["results"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No responses recorded.")

    st.markdown("### Diversity Analysis")

    summary = report.get("summary")

    if isinstance(summary, list) and summary:
        summary_df = pd.DataFrame(summary)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        required_columns = {
            "parameter_value",
            "unique_response_ratio",
        }

        if required_columns.issubset(summary_df.columns):
            chart_df = summary_df.set_index(
                "parameter_value"
            )[["unique_response_ratio"]]

            st.bar_chart(chart_df)

    else:
        st.info(
            "No diversity summary is available."
        )

    st.divider()

    original_report = {
        key: value
        for key, value in report.items()
        if key != "_file_path"
    }

    st.download_button(
        "Download Historical JSON Report",
        data=json.dumps(
            original_report,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8"),
        file_name=Path(
            report["_file_path"]
        ).name,
        mime="application/json",
    )




def render_comparison_tab() -> None:
    """Compare multiple saved sampling experiments."""

    st.subheader("Multi-Experiment Comparison")

    st.caption(
        "Select two or more saved experiments to compare "
        "sampling diversity and experiment configurations."
    )

    reports = [
        report
        for report in list_experiment_reports()
        if (
            report.get("status") == "completed"
            and isinstance(report.get("summary"), list)
            and bool(report["summary"])
            and isinstance(report.get("results"), list)
        )
    ]

    strategies = [
        strategy
        for strategy in ("temperature", "top_p", "top_k")
        if sum(
            report.get("experiment_type") == strategy
            for report in reports
        ) >= 2
    ]

    if not strategies:
        st.info(
            "At least two completed experiments using "
            "the same sampling strategy are required."
        )
        return

    # ----------------------------------------------------
    # STRATEGY SELECTION
    # ----------------------------------------------------

    strategy = st.selectbox(
        "Sampling Strategy",
        options=strategies,
        key="multi_comparison_strategy",
    )

    candidates = [
        report
        for report in reports
        if report.get("experiment_type") == strategy
    ]

    report_by_path = {
        report["_file_path"]: report
        for report in candidates
    }

    paths = list(report_by_path)

    def format_report(path: str) -> str:
        report = report_by_path[path]

        date = str(
            report.get("started_at_utc") or "Unknown date"
        )[:19].replace("T", " ")

        return (
            f"{date} | "
            f"{report.get('provider', 'unknown')} | "
            f"{report.get('model', 'unknown')} | "
            f"{len(report['results'])} responses | "
            f"{Path(path).name}"
        )

    # ----------------------------------------------------
    # MULTIPLE SELECTION
    # ----------------------------------------------------

    selected_paths = st.multiselect(
        "Select experiments to compare",
        options=paths,
        default=paths[:2],
        format_func=format_report,
        key="multi_comparison_selection",
    )

    if len(selected_paths) < 2:
        st.warning(
            "Please select at least two experiments."
        )
        return

    if len(selected_paths) > 8:
        st.info(
            "More than eight experiments are selected. "
            "Consider narrowing your selection "
            "to keep the chart readable."
        )

    selected_reports = [
        report_by_path[path]
        for path in selected_paths
    ]

    # ----------------------------------------------------
    # VALIDATE AND COMPARE
    # ----------------------------------------------------

    try:
        comparison = compare_multiple_experiment_reports(
            selected_reports
        )
    except ValueError as exc:
        st.error(f"Comparison error: {exc}")
        return

    st.divider()

    # ----------------------------------------------------
    # SUMMARY METRICS
    # ----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Experiments",
            len(selected_reports),
        )

    with col2:
        st.metric(
            "Total Responses",
            sum(
                len(report["results"])
                for report in selected_reports
            ),
        )

    with col3:
        st.metric(
            "Sampling Strategy",
            strategy.replace("_", "-"),
        )

    # ----------------------------------------------------
    # EXPERIMENT OVERVIEW
    # ----------------------------------------------------

    st.markdown("### Selected Experiments")

    overview_rows = []

    for index, report in enumerate(
        selected_reports,
        start=1,
    ):
        parameters = report.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}

        overview_rows.append({
            "Experiment": f"Experiment {index}",
            "Provider": report.get("provider", "unknown"),
            "Model": report.get("model", "unknown"),
            "Responses": len(report["results"]),
            "Repetitions": parameters.get(
                "repetitions"
            ),
            "Started": report.get(
                "started_at_utc", "unknown"
            ),
        })

    st.dataframe(
        pd.DataFrame(overview_rows),
        use_container_width=True,
        hide_index=True,
    )

    # ----------------------------------------------------
    # COMPARABILITY WARNINGS
    # ----------------------------------------------------

    st.markdown("### Comparability Checks")

    models = {
        (report.get("provider"), report.get("model"))
        for report in selected_reports
    }

    prompts = {
        report.get("prompt")
        for report in selected_reports
    }

    response_counts = {
        len(report["results"])
        for report in selected_reports
    }

    fixed_parameters = []

    parameter_sets = []

    for report in selected_reports:
        parameters = report.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}

        fixed_parameters.append(
            parameters.get("fixed", {})
        )

        values = parameters.get("values", [])

        if isinstance(values, list):
            parameter_sets.append(
                tuple(sorted(values))
            )
        else:
            parameter_sets.append(())

    warnings = []

    if len(models) > 1:
        warnings.append(
            "Different providers or models are selected."
        )

    if len(prompts) > 1:
        warnings.append(
            "Experiments use different prompts."
        )

    if len(response_counts) > 1:
        warnings.append(
            "Response counts differ. Unique-response "
            "ratios depend on sample size."
        )

    if any(
        fixed != fixed_parameters[0]
        for fixed in fixed_parameters[1:]
    ):
        warnings.append(
            "Fixed sampling parameters differ."
        )

    if len(set(parameter_sets)) > 1:
        warnings.append(
            "Experiments do not all use the same "
            "sampling parameter values."
        )

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "The selected experiments have matching "
            "models, prompts, response counts, fixed "
            "parameters and tested parameter values."
        )

    # ----------------------------------------------------
    # COMPARISON TABLE
    # ----------------------------------------------------

    st.divider()
    st.markdown("### Diversity Comparison")

    comparison_df = pd.DataFrame(comparison)

    experiment_columns = [
        f"Experiment {index}"
        for index in range(1, len(selected_reports) + 1)
    ]

    display_df = comparison_df.copy()

    for column in experiment_columns:
        display_df[column] = (
            pd.to_numeric(
                display_df[column],
                errors="coerce",
            ) * 100
        ).round(1)

    display_df = display_df.rename(
        columns={
            column: f"{column} (%)"
            for column in experiment_columns
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ----------------------------------------------------
    # MULTI-EXPERIMENT CHART
    # ----------------------------------------------------

    st.markdown("### Response Diversity Chart")

    chart_df = comparison_df.copy()

    for column in experiment_columns:
        chart_df[column] = (
            pd.to_numeric(
                chart_df[column],
                errors="coerce",
            ) * 100
        )

    chart_df["Parameter"] = (
        chart_df["Parameter"].astype(str)
    )

    st.bar_chart(
        chart_df.set_index("Parameter")[
            experiment_columns
        ],
        stack=False,
        use_container_width=True,
    )

    st.caption(
        "Unique-response ratios are shown as percentages. "
        "Each experiment is represented by a separate "
        "bar series. Missing values are not treated as zero."
    )

    # ----------------------------------------------------
    # EXPERIMENT CONFIGURATIONS
    # ----------------------------------------------------

    render_comparison_insights(selected_reports)

    with st.expander("View All Experiment Configurations"):
        for index, report in enumerate(
            selected_reports,
            start=1,
        ):
            st.markdown(f"**Experiment {index}**")

            st.json({
                "provider": report.get("provider"),
                "model": report.get("model"),
                "prompt": report.get("prompt"),
                "parameters": report.get("parameters"),
                "started_at_utc": report.get(
                    "started_at_utc"
                ),
            })

    # ----------------------------------------------------
    # CSV EXPORT
    # ----------------------------------------------------

    st.divider()

    st.download_button(
        "Download Comparison CSV",
        data=display_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="multi_experiment_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )



def render_comparison_insights(
    selected_reports: list[dict],
) -> None:
    """Visualize counts and diversity differences."""

    rows = build_comparison_insights(
        selected_reports
    )

    if not rows:
        st.info("No comparison metrics available.")
        return

    data = pd.DataFrame(rows)

    experiment_count = len(selected_reports)

    total_columns = [
        f"E{i} Total"
        for i in range(1, experiment_count + 1)
    ]

    unique_columns = [
        f"E{i} Unique"
        for i in range(1, experiment_count + 1)
    ]

    delta_columns = [
        f"E{i} - E1 (pp)"
        for i in range(2, experiment_count + 1)
    ]

    st.divider()
    st.markdown("### Advanced Comparison Metrics")

    st.dataframe(
        data,
        hide_index=True,
        width="stretch",
    )

    chart_data = data.copy()
    chart_data["Parameter"] = (
        chart_data["Parameter"].astype(str)
    )
    chart_data = chart_data.set_index("Parameter")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Total Responses")

        st.bar_chart(
            chart_data[total_columns],
            stack=False,
            sort=False,
            width="stretch",
        )

    with col2:
        st.markdown("#### Unique Responses")

        st.bar_chart(
            chart_data[unique_columns],
            stack=False,
            sort=False,
            width="stretch",
        )

    st.markdown(
        "### Diversity Difference vs Experiment 1"
    )

    st.caption(
        "Each selected experiment is compared with "
        "Experiment 1 in percentage points. "
        "Positive values indicate higher observed "
        "diversity; negative values indicate lower."
    )

    if delta_columns:
        st.bar_chart(
            chart_data[delta_columns],
            stack=False,
            sort=False,
            width="stretch",
        )



def render_performance_tab() -> None:
    """Visualize latency and token usage metrics."""

    st.subheader("Performance Analytics")

    st.caption(
        "Analyze end-to-end generation latency and "
        "provider-reported token usage."
    )

    source = st.radio(
        "Data source",
        ["Latest Run", "Saved JSON Report"],
        horizontal=True,
        key="performance_source",
    )

    if source == "Latest Run":
        last_run = st.session_state.get("last_run")

        if last_run is None:
            st.info("Run an experiment first.")
            return

        results = last_run["results"]
        strategy = last_run["strategy"]
        model = last_run["model"]

    else:
        reports = [
            report
            for report in list_experiment_reports()
            if report.get("status") == "completed"
        ]

        if not reports:
            st.info("No completed reports available.")
            return

        selected = st.selectbox(
            "Select saved experiment",
            options=list(range(len(reports))),
            format_func=lambda i: (
                f"{str(reports[i].get('started_at_utc', ''))[:19]}"
                f" | {reports[i].get('provider')}"
                f" | {reports[i].get('model')}"
                f" | {reports[i].get('experiment_type')}"
            ),
            key="performance_history_selection",
        )

        report = reports[selected]

        results = report["results"]
        strategy = report["experiment_type"]
        model = report.get("model", "unknown")

    st.caption(
        f"Model: {model} | Strategy: {strategy}"
    )

    try:
        rows = build_performance_rows(
            results,
            strategy,
        )

        summary_rows = summarize_performance_rows(rows)

    except ValueError as exc:
        st.error(f"Performance data error: {exc}")
        return

    if not rows:
        st.info("No request records available.")
        return

    details_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    measured = details_df["Latency (s)"].dropna()

    input_usage = details_df["Input Tokens"].dropna()
    output_usage = details_df["Output Tokens"].dropna()

    # ----------------------------------------------------
    # SUMMARY METRICS
    # ----------------------------------------------------

    st.divider()

    complete_usage = details_df.dropna(
        subset=["Input Tokens", "Output Tokens"]
    )
    complete_total_tokens = (
        complete_usage["Input Tokens"] + complete_usage["Output Tokens"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", len(rows), border=True)

    with col2:
        st.metric(
            "Average Latency",
            f"{measured.mean():.3f} s" if not measured.empty else "N/A",
            border=True,
        )

    with col3:
        st.metric(
            "Recorded Total Tokens",
            f"{int(complete_total_tokens.sum()):,}"
            if not complete_total_tokens.empty else "N/A",
            border=True,
        )

    with col4:
        st.metric(
            "Avg Total Tokens / Request",
            f"{complete_total_tokens.mean():,.2f}"
            if not complete_total_tokens.empty else "N/A",
            border=True,
        )

    in_total, in_average, out_total, out_average = st.columns(4)

    in_total.metric(
        "Input Tokens (reported)",
        f"{int(input_usage.sum()):,}" if not input_usage.empty else "N/A",
        border=True,
    )
    in_average.metric(
        "Avg Input Tokens / Request",
        f"{input_usage.mean():,.2f}" if not input_usage.empty else "N/A",
        border=True,
    )
    out_total.metric(
        "Output Tokens (reported)",
        f"{int(output_usage.sum()):,}" if not output_usage.empty else "N/A",
        border=True,
    )
    out_average.metric(
        "Avg Output Tokens / Request",
        f"{output_usage.mean():,.2f}" if not output_usage.empty else "N/A",
        border=True,
    )
    st.caption(
        "Token coverage — input: "
        f"{len(input_usage)}/{len(rows)}, output: "
        f"{len(output_usage)}/{len(rows)}, both: "
        f"{len(complete_usage)}/{len(rows)} requests. "
        "Averages divide by requests with the relevant token measurements."
    )

    if measured.empty:
        st.info(
            "This experiment has no recorded latency "
            "measurements. Older JSON reports may not "
            "contain benchmark fields."
        )

    if (
        len(input_usage) < len(rows)
        or len(output_usage) < len(rows)
    ):
        st.warning(
            "Token usage is missing for some requests. "
            "Totals and averages use only provider-reported values; "
            "the combined total uses requests with both counts."
        )

    # ----------------------------------------------------
    # GROUPED PERFORMANCE SUMMARY
    # ----------------------------------------------------

    st.divider()
    st.markdown("### Performance by Sampling Parameter")

    st.dataframe(
        summary_df,
        hide_index=True,
        width="stretch",
    )

    chart_df = summary_df.copy()
    chart_df["Parameter"] = (
        chart_df["Parameter"].astype(str)
    )
    chart_df = chart_df.set_index("Parameter")

    # ----------------------------------------------------
    # LATENCY COMPARISON
    # ----------------------------------------------------

    st.markdown("### Average vs Median Latency")

    latency_columns = [
        "Average Latency (s)",
        "Median Latency (s)",
        "P95 Latency (s)",
    ]

    if chart_df[latency_columns].notna().any().any():
        st.bar_chart(
            chart_df[latency_columns],
            stack=False,
            sort=False,
            width="stretch",
        )
    else:
        st.info("No latency measurements to chart.")

    # ----------------------------------------------------
    # TOKEN USAGE
    # ----------------------------------------------------

    st.markdown("### Average Token Usage per Request")

    average_token_columns = [
        "Average Input Tokens / Request",
        "Average Output Tokens / Request",
        "Average Total Tokens / Request",
    ]

    if chart_df[average_token_columns].notna().any().any():
        st.bar_chart(
            chart_df[average_token_columns],
            stack=False,
            sort=False,
            width="stretch",
        )
    else:
        st.info("No token usage data available.")

    with st.expander("View absolute token totals by parameter"):
        total_token_columns = [
            "Recorded Input Tokens",
            "Recorded Output Tokens",
            "Recorded Total Tokens",
        ]
        st.bar_chart(
            chart_df[total_token_columns],
            stack=False,
            sort=False,
            width="stretch",
        )
    st.caption(
        "Per-request averages use requests with recorded token usage. "
        "Total token counts rise with request count."
    )

    # ----------------------------------------------------
    # INDIVIDUAL REQUEST LATENCY
    # ----------------------------------------------------

    st.markdown("### Request Latency Timeline")

    timeline_df = details_df[
        ["Request", "Latency (s)"]
    ].copy()

    timeline_df = timeline_df.dropna(
        subset=["Latency (s)"]
    )

    if not timeline_df.empty:
        st.line_chart(
            timeline_df.set_index("Request"),
            width="stretch",
        )
    else:
        st.info("No request latency data available.")

    # ----------------------------------------------------
    # RAW REQUEST METRICS
    # ----------------------------------------------------

    st.markdown("### Individual Request Metrics")

    st.dataframe(
        details_df,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download Performance CSV",
        data=details_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="performance_metrics.csv",
        mime="text/csv",
    )

    st.caption(
        "Latency includes provider request overhead "
        "and may include initial model loading. "
        "Token counts are reported by the provider."
    )



def render_benchmark_comparison_tab() -> None:
    """Compare benchmark performance across saved experiments."""

    st.subheader("Benchmark Comparison")

    st.caption(
        "Compare latency, token usage and experiment "
        "conditions across two or more saved reports."
    )

    reports = [
        report
        for report in list_experiment_reports()
        if (
            report.get("status") == "completed"
            and isinstance(report.get("results"), list)
            and report.get("experiment_type")
            in {"temperature", "top_p", "top_k"}
        )
    ]

    strategies = [
        strategy
        for strategy in ("temperature", "top_p", "top_k")
        if sum(
            report.get("experiment_type") == strategy
            for report in reports
        ) >= 2
    ]

    if not strategies:
        st.info(
            "At least two completed experiments using "
            "the same sampling strategy are required."
        )
        return

    strategy = st.selectbox(
        "Sampling strategy",
        options=strategies,
        format_func=lambda value: value.replace("_", "-"),
        key="benchmark_strategy",
    )

    candidates = [
        report
        for report in reports
        if report.get("experiment_type") == strategy
    ]

    def format_report(index: int) -> str:
        report = candidates[index]

        date = str(
            report.get("started_at_utc") or "Unknown date"
        )[:19].replace("T", " ")

        filename = Path(
            str(report.get("_file_path") or "report.json")
        ).name

        return (
            f"{date} | "
            f"{report.get('provider', 'unknown')} | "
            f"{report.get('model', 'unknown')} | "
            f"{len(report['results'])} requests | "
            f"{filename}"
        )

    selected_indices = st.multiselect(
        "Select experiments",
        options=list(range(len(candidates))),
        default=list(range(min(2, len(candidates)))),
        format_func=format_report,
        key=f"benchmark_selection_{strategy}",
    )

    if len(selected_indices) < 2:
        st.warning("Select at least two experiments.")
        return

    selected_reports = [
        candidates[index]
        for index in selected_indices
    ]

    try:
        comparison = compare_benchmark_reports(
            selected_reports
        )
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Benchmark comparison failed: {exc}")
        return

    summary_df = pd.DataFrame(
        comparison["summary_rows"]
    )

    parameter_df = pd.DataFrame(
        comparison["parameter_rows"]
    )

    st.divider()

    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Experiments",
            comparison["experiment_count"],
            border=True,
        )

    with col2:
        st.metric(
            "Total Requests",
            int(summary_df["Requests"].sum()),
            border=True,
        )

    with col3:
        st.metric(
            "Measured Requests",
            int(summary_df["Measured Requests"].sum()),
            border=True,
        )

    # --------------------------------------------------
    # COMPARABILITY CHECKS
    # --------------------------------------------------

    st.markdown("### Comparability Checks")

    warnings = list(comparison["warnings"])

    provider_models = {
        (
            report.get("provider"),
            report.get("model"),
        )
        for report in selected_reports
    }

    if len(provider_models) > 1:
        warnings.insert(
            0,
            "Different providers or models are selected. "
            "Latency and token counts may not be "
            "directly comparable."
        )

    order_seeds = {
        (report.get("parameters") or {}).get("order_seed")
        for report in selected_reports
    }

    if len(order_seeds) > 1:
        warnings.append(
            "Execution order seeds differ."
        )

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "No comparability differences detected "
            "in the checked experiment settings."
        )

    # --------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------

    st.divider()
    st.markdown("### Experiment Benchmark Summary")

    st.dataframe(
        summary_df,
        hide_index=True,
        width="stretch",
    )

    chart_df = summary_df.set_index(
        "Experiment"
    ).copy()

    # --------------------------------------------------
    # LATENCY COMPARISON
    # --------------------------------------------------

    st.markdown("### Latency Comparison")

    latency_columns = [
        "Average Latency (s)",
        "Median Latency (s)",
        "P95 Latency (s)",
    ]

    latency_df = chart_df[
        latency_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if latency_df.notna().any().any():
        st.bar_chart(
            latency_df,
            stack=False,
            width="stretch",
        )
    else:
        st.info(
            "Selected reports have no latency measurements."
        )

    # --------------------------------------------------
    # TOKEN COMPARISON
    # --------------------------------------------------

    st.markdown("### Average Token Usage per Request")

    average_columns = [
        "Average Input Tokens / Request",
        "Average Output Tokens / Request",
        "Average Total Tokens / Request",
    ]
    average_df = chart_df[average_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if average_df.notna().any().any():
        st.bar_chart(
            average_df,
            stack=False,
            width="stretch",
        )
    else:
        st.info("Selected reports have no token usage data.")

    st.markdown("#### Token Usage — Numeric Comparison")
    token_summary = pd.DataFrame(
        build_token_comparison_rows(comparison["summary_rows"])
    )
    st.dataframe(
        token_summary,
        column_config={
            "Provider / Model": st.column_config.TextColumn(
                "Provider / Model"
            ),
            "Requests": st.column_config.NumberColumn(
                "Requests", format="%d"
            ),
            "Recorded Total Tokens": st.column_config.NumberColumn(
                "Recorded Total Tokens", format="%d"
            ),
            "Average Total Tokens / Request": st.column_config.NumberColumn(
                "Avg Tokens / Request", format="%.2f"
            ),
            "Difference vs E1": st.column_config.TextColumn(
                "Avg Token Difference vs E1 (%)"
            ),
            "Token Coverage": st.column_config.TextColumn(
                "Token Coverage (input + output)"
            ),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "E1 is the first selected experiment (reference). A positive "
        "percentage means higher average tokens per request than E1; "
        "a negative percentage means lower usage. N/A means the reference "
        "has zero or missing average usage, or the comparison is missing."
    )

    with st.expander("View absolute token totals"):
        token_columns = [
            "Recorded Input Tokens",
            "Recorded Output Tokens",
            "Recorded Total Tokens",
        ]
        totals_df = chart_df[token_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if totals_df.notna().any().any():
            st.bar_chart(
                totals_df,
                stack=False,
                width="stretch",
            )
        else:
            st.info("Selected reports have no token totals.")

    st.caption(
        "Average tokens per request uses only requests with the "
        "respective reported token counts. Combined totals and averages "
        "use requests with both input and output usage. "
        "See Input/Output/Total Token Coverage in the table above. "
        "Provider tokenizers may differ, so cross-provider numbers "
        "are not necessarily directly comparable."
    )

    # --------------------------------------------------
    # PARAMETER-LEVEL METRICS
    # --------------------------------------------------

    st.divider()
    st.markdown("### Metrics by Sampling Parameter")

    st.dataframe(
        parameter_df,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------
    # EXPORTS
    # --------------------------------------------------

    st.divider()
    st.markdown("### Export Benchmark Report")

    export_data = {
        "experiment_count": comparison["experiment_count"],
        "metadata": [
            report["metadata"]
            for report in comparison["reports"]
        ],
        "summary_rows": comparison["summary_rows"],
        "parameter_rows": comparison["parameter_rows"],
        "warnings": warnings,
    }

    csv_col, json_col = st.columns(2)

    with csv_col:
        st.download_button(
            "Download Benchmark CSV",
            data=summary_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="benchmark_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with json_col:
        st.download_button(
            "Download Benchmark JSON",
            data=json.dumps(
                export_data,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name="benchmark_comparison.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(
        "P95 estimates from small samples can be unstable. "
        "Performance differences do not by themselves "
        "indicate differences in generation quality."
    )



def render_controlled_benchmark_tab(
    config: dict,
    settings: dict,
) -> None:
    """Configure, preview and run controlled benchmarks."""

    st.subheader("Controlled Benchmark")

    st.caption(
        "Run a standardized benchmark with warm-up, "
        "reproducible execution ordering and "
        "measured performance metrics."
    )

    provider_name = config["provider"]
    model = config["model"].strip()
    prompt = config["prompt"].strip()

    st.write(
        f"**Provider:** {provider_name} | "
        f"**Model:** {model}"
    )

    strategies = ["temperature", "top_p"]

    if provider_name == "ollama":
        strategies.append("top_k")

    strategy = st.selectbox(
        "Benchmark strategy",
        strategies,
        format_func=lambda value: value.replace("_", "-"),
        key="controlled_strategy",
    )

    if strategy == "temperature":
        values = tuple(
            settings["temperature"]["values"]
        )
        fixed_temperature = 0.8
        fixed_top_p = float(
            settings["temperature"]["top_p"]
        )

    elif strategy == "top_p":
        values = tuple(
            settings["top_p"]["values"]
        )
        fixed_temperature = float(
            settings["top_p"]["temperature"]
        )
        fixed_top_p = 0.9

    else:
        values = tuple(TOP_K_VALUES)
        fixed_temperature = TOP_K_TEMPERATURE
        fixed_top_p = TOP_K_TOP_P

    st.write("**Sampling values:**", list(values))

    col1, col2 = st.columns(2)

    with col1:
        repetitions = int(
            st.number_input(
                "Repetitions per value",
                min_value=1,
                max_value=100,
                value=3,
                step=1,
                key="controlled_repetitions",
            )
        )

        warmup = int(
            st.selectbox(
                "Warm-up requests",
                options=[0, 1, 2, 3],
                index=1,
                key="controlled_warmup",
            )
        )

    with col2:
        max_tokens = int(
            st.number_input(
                "Maximum output tokens",
                min_value=1,
                max_value=4096,
                value=128,
                step=1,
                key="controlled_max_tokens",
            )
        )

        randomize = st.checkbox(
            "Randomize execution order",
            value=True,
            key="controlled_randomize",
        )

    order_seed = None

    if randomize:
        order_seed = int(
            st.number_input(
                "Execution order seed",
                min_value=0,
                value=42,
                step=1,
                key="controlled_order_seed",
            )
        )

    use_generation_seed = st.checkbox(
        "Use generation seed",
        value=False,
        disabled=provider_name != "ollama",
        key="controlled_use_seed",
    )

    generation_seed = None

    if use_generation_seed and provider_name == "ollama":
        generation_seed = int(
            st.number_input(
                "Generation seed",
                min_value=0,
                value=42,
                step=1,
                key="controlled_generation_seed",
            )
        )

        st.warning(
            "Using the same generation seed may "
            "reduce observed response diversity."
        )

    try:
        benchmark_config = ControlledBenchmarkConfig(
            strategy=strategy,
            prompt=prompt,
            values=values,
            repetitions=repetitions,
            fixed_temperature=fixed_temperature,
            fixed_top_p=fixed_top_p,
            max_output_tokens=max_tokens,
            warmup_requests=warmup,
            randomize_order=randomize,
            order_seed=order_seed,
            generation_seed=generation_seed,
        )

        preview = benchmark_config.preview()

    except ValueError as exc:
        st.error(f"Invalid benchmark settings: {exc}")
        return

    # --------------------------------------------------
    # EXECUTION PREVIEW
    # --------------------------------------------------

    st.divider()
    st.markdown("### Execution Preview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Measured Requests",
            preview["recorded_requests"],
            border=True,
        )

    with c2:
        st.metric(
            "Warm-up Requests",
            preview["warmup_requests"],
            border=True,
        )

    with c3:
        st.metric(
            "Total Provider Calls",
            preview["total_provider_calls"],
            border=True,
        )

    with st.expander("View execution order"):
        st.dataframe(
            pd.DataFrame(
                preview["execution_order"]
            ),
            hide_index=True,
            width="stretch",
        )

    if preview["recorded_requests"] < 20:
        st.info(
            "This is a small benchmark. "
            "P95 estimates may be unstable."
        )

    requires_confirmation = (
        provider_name != "ollama"
        and preview["total_provider_calls"] > 20
    )

    confirmed = True

    if requires_confirmation:
        st.warning(
            "This benchmark uses a cloud provider. "
            "All requests, including warm-up, "
            "may incur API charges."
        )

        confirmed = st.checkbox(
            "I understand the API request volume.",
            key="controlled_cost_confirmation",
        )

    # --------------------------------------------------
    # EXECUTION
    # --------------------------------------------------

    if not st.button(
        "Run Controlled Benchmark",
        type="primary",
        disabled=not confirmed,
        key="controlled_run_button",
    ):
        return

    total = preview["recorded_requests"]
    completed = 0

    progress = st.progress(
        0.0,
        text="Starting controlled benchmark...",
    )

    def update_progress(result: SamplingResult) -> None:
        nonlocal completed

        completed += 1

        progress.progress(
            completed / total,
            text=f"Completed {completed}/{total} requests",
        )

    try:
        with st.spinner("Running controlled benchmark..."):
            with provider_session(provider_name) as provider:
                outcome = run_controlled_benchmark(
                    provider=provider,
                    model=model,
                    config=benchmark_config,
                    on_result=update_progress,
                )

    except Exception as exc:
        st.error(
            "Controlled benchmark failed: "
            f"{type(exc).__name__}: {exc}"
        )
        st.info(
            "If a report was created, completed requests "
            "are retained in Experiment History."
        )
        return

    display_strategy = strategy.replace("_", "-")

    st.session_state["last_run"] = {
        "provider": provider_name,
        "model": model,
        "strategy": display_strategy,
        "prompt": prompt,
        "results": outcome.results,
        "report_path": str(outcome.report_path),
        "warmup_requests": warmup,
        "seed": generation_seed,
        "randomize_order": randomize,
        "order_seed": order_seed,
    }

    st.success("Controlled benchmark completed successfully!")

    metrics = outcome.benchmark_report["metrics"]

    st.markdown("### Benchmark Results")

    request_count, combined_total, combined_average = st.columns(3)
    request_count.metric("Completed Requests", metrics["Requests"])
    combined_total.metric(
        "Recorded Total Tokens",
        metrics["Recorded Total Tokens"]
        if metrics["Recorded Total Tokens"] is not None else "N/A",
    )
    combined_average.metric(
        "Avg Total Tokens / Request",
        metrics["Average Total Tokens / Request"]
        if metrics["Average Total Tokens / Request"] is not None else "N/A",
    )
    st.caption(
        "Combined token coverage: "
        f"{metrics['Total Token Coverage']}/{metrics['Requests']} requests."
    )

    st.dataframe(
        pd.DataFrame(
            outcome.benchmark_report["by_parameter"]
        ),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        f"Saved report: {outcome.report_path}"
    )

    st.download_button(
        "Download Benchmark JSON",
        data=outcome.report_path.read_bytes(),
        file_name=outcome.report_path.name,
        mime="application/json",
        key="controlled_download_json",
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    """Main Streamlit application."""

    try:

        settings = get_settings()

    except Exception as exc:

        st.error(
            "Configuration error: "
            f"{type(exc).__name__}: {exc}"
        )

        st.stop()

    # Sidebar
    config = render_sidebar(settings)

    # Dashboard header
    render_header(config)

    # Execute selected experiment
    handle_experiment_run(
        config=config,
        settings=settings,
    )

    st.divider()

    # Dashboard tabs
    experiment_tab, results_tab, analytics_tab, history_tab, comparison_tab, performance_tab, controlled_tab, = st.tabs(
        [
            "Experiment",
            "Results",
            "Analytics",
            "History",
            "Comparison",
            "Performance",
            "Controlled Benchmark",
        ]
    )

    with experiment_tab:

        render_experiment_tab(config)

    with results_tab:

        render_results_tab()

    with analytics_tab:

        render_analytics_tab()

    with history_tab:

        render_history_tab()

    with comparison_tab:

        render_comparison_tab()

    with performance_tab:
        request_tab, benchmark_tab = st.tabs(
            [
                "Request Analytics",
                "Benchmark Comparison",
            ]
        )

        with request_tab:
            render_performance_tab()

        with benchmark_tab:
            render_benchmark_comparison_tab()

    with controlled_tab:
        render_controlled_benchmark_tab(
            config=config,
            settings=settings,
        )

if __name__ == "__main__":
    main()