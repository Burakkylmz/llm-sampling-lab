# LLM Sampling Lab

**A hands-on laboratory for LLM sampling, reproducibility, and performance evaluation.**

LLM Sampling Lab is a Python project for investigating how **Temperature**, **Top-p (nucleus sampling)** and **Top-k** affect probability distributions and text generation. It combines real model experiments, an offline token-sampling simulator, and a **Streamlit dashboard** for interactive experiments and comparative benchmarking.

## Quick start: Docker web interface

**Requirements:** Git and Docker Desktop (Windows/macOS) or Docker Engine with Docker Compose (Linux). Python and uv are **not** required on the host.

~~~bash
git clone https://github.com/Burakkylmz/llm-sampling-lab.git
cd llm-sampling-lab
docker compose up --build -d
~~~

Open **http://localhost:8501** in your browser.

The UI starts without cloud API keys. To make live Groq or OpenAI requests, create your own private **.env** file:

**Windows PowerShell:**

~~~powershell
Copy-Item .env.example .env
~~~

**macOS / Linux:**

~~~bash
cp .env.example .env
~~~

Edit the file and add the keys you need:

~~~dotenv
GROQ_API_KEY=
OPENAI_API_KEY=
STREAMLIT_PORT=8501
~~~

After editing **.env**, recreate the UI container:

~~~bash
docker compose up -d --force-recreate ui
~~~

Useful management commands:

~~~bash
docker compose ps
docker compose logs -f ui
docker compose down
~~~

The dashboard is published to **127.0.0.1** on the host by default. You can change the host port with **STREAMLIT_PORT**. Experiment reports remain in the host's **results/** directory after containers are stopped or recreated.

The original CLI continues to work:

~~~bash
docker compose run --rm sampling-lab --help
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
~~~

**More setup information:** [Docker Web UI Guide](docs/docker-ui.md), including Docker Desktop, Ollama networking, troubleshooting, and port configuration.

---

## Features

| Area | Capabilities |
| --- | --- |
| **Streamlit UI** | Configure experiments, view results, run analytics, browse history, compare experiments, and execute controlled benchmarks |
| **Real LLM experiments** | Temperature and Top-p with Groq, OpenAI, and Ollama; native Top-k with Ollama |
| **Offline simulator** | Temperature, Top-k and Top-p over a fixed probability distribution without API requests |
| **Movie review generator** | Local Temperature and Top-k select a review style; Groq generates Turkish reviews |
| **Performance metrics** | Request latency, input/output token usage, per-request averages, measurement coverage |
| **Benchmark comparison** | Cross-experiment comparisons, mean/median/P95 latency, token consumption and baseline differences |
| **Reproducibility** | Repetition counts, warm-up requests, separate execution-order seeds, supported generation seeds |
| **Persistence** | Incremental JSON reports, CSV/JSON exports from supported dashboard views, and PNG plots |

The dashboard includes **Experiment**, **Results**, **Analytics**, **History**, **Comparison**, **Performance**, and **Controlled Benchmark** tabs. The Performance tab contains **Request Analytics** and **Benchmark Comparison**.

### Provider support

| Provider | Temperature | Top-p | Native Top-k | Required setup |
| --- | --- | --- | --- | --- |
| **Groq** | Model-dependent | Model-dependent | No | GROQ_API_KEY |
| **OpenAI** | Model-dependent | Model-dependent | No | OPENAI_API_KEY |
| **Ollama** | Model-dependent | Model-dependent | Yes, with suitable models | Ollama service and downloaded model |

**Important:** The implemented Groq and OpenAI adapters do **not** expose native Top-k. In the offline simulator, Top-k is applied locally to an educational distribution. In the movie review experiment, Top-k chooses a **review style** locally; it does not change Groq's token sampling. Native Top-k is available for compatible **Ollama** models.

Use model discovery and capability checks before assuming a parameter is supported. The repository's default Groq model is **openai/gpt-oss-20b**; for OpenAI and Ollama CLI experiments, provide an explicit model identifier.

### Ollama on the host, UI in Docker

The Docker setup configures the Ollama endpoint as **http://host.docker.internal:11434**. It must point to an accessible Ollama server running on your computer, not the loopback address inside the container.

For example:

~~~bash
ollama pull llama3.2:1b
ollama list
~~~

If model discovery fails, verify the Ollama service is running and reachable from Docker. You may need to adjust the Ollama server's listening address, while ensuring it is **not exposed to untrusted networks**. The CLI and provider code read the **OLLAMA_HOST** environment variable, and direct local runs default to **http://127.0.0.1:11434**.

For platform-specific details, see [Docker Web UI Guide](docs/docker-ui.md).

---

## Running experiments

Run commands below from the repository root.

### Offline sampling simulations

No API key or external model is required.

The initial token probabilities are:

| Token | Probability |
| --- | ---: |
| AI | 0.40 |
| Machine | 0.25 |
| Deep | 0.15 |
| Neural | 0.12 |
| Data | 0.08 |

~~~bash
docker compose run --rm sampling-lab --simulate temperature --samples 10000 --seed 42
docker compose run --rm sampling-lab --simulate top-k --samples 10000 --seed 42
docker compose run --rm sampling-lab --simulate top-p --samples 10000 --seed 42
~~~

The simulator applies each configured parameter value, samples token indices with a seeded random generator, and compares theoretical probabilities with observed frequencies. PNG charts are saved in **results/visualizations/**.

Temperature below 1 concentrates probability on more likely candidates; temperature above 1 flattens the distribution. Top-k retains a fixed number of candidates. Top-p retains the smallest top-probability prefix that reaches the cumulative threshold.

### Live Temperature and Top-p

~~~bash
# Groq (default provider)
docker compose run --rm sampling-lab --experiment temperature --repetitions 3
docker compose run --rm sampling-lab --experiment top-p --repetitions 3

# OpenAI (requires an API key and compatible model)
docker compose run --rm sampling-lab --provider openai --model gpt-4.1-mini --experiment temperature --repetitions 3
~~~

These experiments vary one parameter while holding the other fixed. The CLI uses an electric-car slogan prompt; the dashboard allows a custom prompt.

### Native Top-k (Ollama)

~~~bash
docker compose run --rm sampling-lab --provider ollama --model llama3.2:1b --experiment top-k --repetitions 3
~~~

The preset Top-k values are **1, 20, 40**. This experiment forwards Top-k to the local Ollama model rather than applying it only to a simulated distribution.

### Movie review experiment (Groq only)

~~~bash
docker compose run --rm sampling-lab --experiment movie-reviews
~~~

The experiment uses the fictional film **Son İstasyon**. Local Temperature and Top-k choose a positive, critical, technical, or emotional style; Groq produces a Turkish review.

Basic checks verify the film title and an 80–100-word length. These structural checks do **not** establish factual accuracy, fluency, or semantic adherence.

### Model and capability commands

~~~bash
docker compose run --rm sampling-lab --provider groq --list-models
docker compose run --rm sampling-lab --provider ollama --discover-models
docker compose run --rm sampling-lab --provider ollama --model llama3.2:1b --show-capabilities
docker compose run --rm sampling-lab --help
~~~

The CLI actions for simulation, experiments, API checks, model listing, discovery, and capability display are mutually exclusive. Use **--samples** and **--seed** only with offline simulation; **--repetitions** applies to real sampling experiments, not the movie-review CLI.

---

## Understanding the benchmark dashboard

**Performance → Request Analytics** displays request-level metrics and aggregates by sampling parameter.

**Performance → Benchmark Comparison** compares saved experiments of the same strategy. The numeric comparison table includes the provider/model, number of requests, reported token totals, **average tokens per request**, measurement coverage, and percentage difference from the first selected experiment (E1).

| Measurement | Meaning |
| --- | --- |
| Mean / median latency | Central tendency of measured request duration |
| P95 latency | Approximate upper-tail duration; can be unstable for small samples |
| Total tokens | Token consumption summed over requests with measured usage |
| **Average tokens / request** | Reported token usage divided by the number of requests with that usage measurement |
| Coverage | Number of requests with usage data out of total recorded requests |
| Unique response ratio | Number of distinct exact response strings divided by total responses |

A 20-request experiment will often use more **total** tokens than a 10-request experiment. That alone does not demonstrate lower token efficiency. Compare per-request averages **and** measurement coverage. Input+output totals and averages are computed only when **both** measurements exist for a request; missing usage is not treated as zero.

The percentage difference against E1 uses:

~~~text
Difference (%) = (Experiment average - E1 average) / E1 average × 100
~~~

A positive result means higher average consumption than E1; a negative result means lower. Missing or zero baseline values are displayed as **N/A**. Different providers may use different tokenizers, so cross-provider token counts are not necessarily directly comparable.

**Controlled Benchmark** provides configurable repetitions, warm-ups, random execution order, and supported seeds. Warm-up calls are excluded from measured results. Seeds help reproducibility but do not guarantee identical responses from external APIs.

**Interpretation limits:** The benchmark measures usage, timing and exact-text diversity; it does not automatically grade response quality, semantic diversity or monetary cost.

---

## Configuration and saved reports

The main configuration is **settings.toml**. Its default sections include:

~~~toml
[groq]
model = "openai/gpt-oss-20b"

[experiments]
repetitions = 5

[temperature]
values = [0.1, 0.5, 1.0]
top_p = 1.0

[top_p]
values = [0.3, 0.7, 1.0]
temperature = 0.7

[local_sampling]
top_k_values = [1, 3, 5]
top_p_values = [0.3, 0.7, 0.9]
temperature_values = [0.5, 1.0, 1.5]
~~~

Native Ollama Top-k and movie-review experiments have additional scenario defaults in their respective Python modules.

Saved results follow this structure:

~~~text
results/
├── temperature/       # Real experiments
├── top_p/             # Real experiments
├── top_k/             # Native Ollama experiments
├── movie_reviews/     # Review reports
└── visualizations/    # Offline sampling PNG charts
~~~

Experiment JSON records are written incrementally with **running**, **completed**, or **failed** statuses. Reports remain available if a later API request fails; an unexpected process exit may leave a report marked **running**.

In Docker Compose, **./results** is mounted to **/app/results** and **settings.toml** is mounted read-only. Data remains on the host when containers are recreated.

### Project layout

~~~text
llm-sampling-lab/
├── .env.example
├── .streamlit/config.toml
├── Dockerfile
├── docker-compose.yaml
├── streamlit_app.py
├── settings.toml
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── docker-ui.md
│   └── participant-quick-start.md
├── src/llm_sampling_lab/
│   ├── cli.py
│   ├── sampling.py
│   ├── simulator.py
│   ├── results.py
│   ├── providers/
│   └── services/
├── tests/
└── results/             # Generated locally; excluded from Git
~~~

---

## Local development without Docker

Requires **Python 3.12+** and **uv**:

~~~bash
uv sync
uv run streamlit run streamlit_app.py
uv run sampling-lab --simulate top-k
uv run pytest -q
~~~

Docker Compose automatically reads an optional **.env** file for variable substitution. Direct **uv run** commands do not automatically load it; to use cloud providers locally:

~~~bash
uv run --env-file .env streamlit run streamlit_app.py
~~~

## Testing and troubleshooting

~~~bash
# Python regression tests
uv run pytest -q

# Docker configuration and web service
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs --tail=100 ui
~~~

| Symptom | What to check |
| --- | --- |
| Docker cannot connect | Docker Desktop / Docker Engine is running |
| Browser page does not load | Compose service health and UI logs |
| Port 8501 is occupied | Change **STREAMLIT_PORT** in **.env**, recreate the UI |
| API authentication fails | Key configured in private **.env** and container recreated |
| Ollama connection refused | Ollama is running, model installed, and endpoint reachable from Docker |
| No historic data appears | Local **results/** directory and Compose volume mount |
| Unsupported sampling parameter | Selected provider/model capabilities |

The Docker UI guide includes expanded Windows and Ollama troubleshooting.

## Security and limitations

- The Streamlit UI is intended for local use and does not implement production authentication.
- Keep credentials out of source control, shared logs, screenshots and issue reports.
- Cloud API experiments may incur provider fees or rate limits.
- Never expose an unauthenticated Ollama server to untrusted networks.
- Offline simulations use a fixed educational distribution; real autoregressive models recompute probabilities at each generation step.
- Sampling diversity and resource consumption are not substitutes for evaluation of response correctness.
- The repository currently contains **no explicit open-source license**; public visibility is not permission to redistribute the source.

## Documentation

- [Docker Web UI Guide](docs/docker-ui.md)
- [Participant Quick Start](docs/participant-quick-start.md)
- [GitHub Issues](https://github.com/Burakkylmz/llm-sampling-lab/issues)
