
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
    )

    try:
        with provider_session(provider_name) as provider:
            service = ExperimentService(
                provider=provider,
                model=model,
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
            )

        st.session_state["last_run"] = {
            "provider": config["provider"],
            "model": config["model"].strip(),
            "strategy": config["experiment"],
            "prompt": config["prompt"].strip(),
            "results": results,
            "report_path": report_path,
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
    experiment_tab, results_tab, analytics_tab, history_tab, comparison_tab, = st.tabs(
        [
            "Experiment",
            "Results",
            "Analytics",
            "History",
            "Comparison",
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


if __name__ == "__main__":
    main()