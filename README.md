# LLM Sampling Lab

**An educational laboratory for observing how temperature, top-k, and top-p affect token selection.**

The project combines two complementary approaches:

1. **Real LLM experiments:** run controlled temperature and top-p experiments through the Groq Chat Completions API and persist each successful response in JSON.
2. **Offline token simulation:** transform a fixed educational token distribution, sample from it, and generate PNG charts comparing theoretical and observed probabilities.

> **Scope:** This lab teaches sampling behavior. It does not benchmark token consumption, inference latency, throughput, or monetary cost. Those topics are intentionally reserved for a separate lab.

## What you will learn

- How temperature sharpens or flattens a probability distribution.
- How top-k retains a fixed number of high-probability tokens.
- How top-p (nucleus sampling) retains the smallest sorted prefix whose cumulative probability reaches the threshold.
- Why theoretical probability and empirical sampling frequency are not identical.
- Why exact-text diversity is only a basic proxy for linguistic diversity.
- How to run repeatable offline experiments and save results.

## Quick start: Docker (recommended)

**Required:** Docker Desktop or a functioning Docker Engine with Docker Compose support. **Not required:** a local Python or `uv` installation.

From the repository root, build the image:

```shell
docker compose build
```

Run an offline experiment; **no API key is needed**:

```shell
docker compose run --rm sampling-lab --simulate temperature --samples 1000 --seed 42
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
docker compose run --rm sampling-lab --simulate top-p --samples 1000 --seed 42
```

PNG charts are saved on your computer under `results/visualizations/` because Docker Compose mounts the project `results/` directory into the container.

To run a **real Groq API experiment**, obtain your own `GROQ_API_KEY`. Create `.env` in the repository root (you can copy `.env.example` if present), and enter:

```dotenv
GROQ_API_KEY=your_own_groq_api_key_here
```

Then run:

```shell
docker compose run --rm sampling-lab --check-api
docker compose run --rm sampling-lab --experiment temperature --repetitions 1
docker compose run --rm sampling-lab --experiment top-p --repetitions 1
```

The JSON reports are saved under `results/temperature/` and `results/top_p/`. **Never commit `.env`, share API keys, or paste secrets into screenshots or logs.**

For complete student instructions, see [Participant Quick Start](docs/participant-quick-start.md).

## Local development with Python and uv

Local installation is optional for participants but useful for developers.

**Requirements:** Python 3.12 and [uv](https://docs.astral.sh/uv/).

```shell
uv sync
uv run sampling-lab --help
uv run pytest -q
```

Offline commands:

```shell
uv run sampling-lab --simulate temperature --samples 10000 --seed 42
uv run sampling-lab --simulate top-k --samples 10000 --seed 42
uv run sampling-lab --simulate top-p --samples 10000 --seed 42
```

For real API experiments, set `GROQ_API_KEY` in your terminal environment before running the commands below. A project `.env` file used by Compose is **not necessarily loaded by direct `uv run` commands**.

```shell
uv run sampling-lab --check-api
uv run sampling-lab --experiment temperature --repetitions 1
uv run sampling-lab --experiment top-p --repetitions 1
```

## CLI reference

| Option | Purpose | API key required? |
| --- | --- | --- |
| `--help` | Display available options | No |
| `--simulate temperature` | Offline temperature simulation | No |
| `--simulate top-k` | Offline top-k simulation | No |
| `--simulate top-p` | Offline top-p simulation | No |
| `--samples N` | Number of offline samples, used with `--simulate` | No |
| `--seed N` | Non-negative random seed, used with `--simulate` | No |
| `--check-api` | Test Groq connectivity | Yes |
| `--experiment temperature` | Real temperature experiment | Yes |
| `--experiment top-p` | Real top-p experiment | Yes |
| `--repetitions N` | Positive request count per parameter value | For real experiments |

`--check-api`, `--experiment`, and `--simulate` are mutually exclusive. Use `--samples` and `--seed` only with `--simulate`.

## Experiment configuration

The file `settings.toml` determines the Groq model and parameter values. The current teaching configuration uses:

```toml
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
```

With Docker Compose, `settings.toml` is mounted read-only into the container. Change it on the host to adjust experiments. Run a small number of repetitions first because real experiments send API requests.

## Outputs and interpretation

```text
results/
├── temperature/       # Real temperature experiments (JSON)
├── top_p/             # Real top-p experiments (JSON)
└── visualizations/    # Offline comparisons (PNG)
```

**JSON report:** includes experiment ID, model, prompt, parameters, UTC timestamps, status, individual responses, and per-parameter diversity summary. Reports are updated after each successful response. If an API error is caught, existing responses remain and the report is marked `failed`; a forcibly interrupted process may leave it `running`.

**Unique Response Ratio (URR):** `distinct exact response strings / total responses`. A value of `1.0` with only one repetition is trivial, not evidence of high diversity. Texts with the same meaning but different wording are treated as distinct.

**Offline PNG:** compares the transformed theoretical distribution with empirical frequencies from repeated random samples. `seed` helps reproduce results in the same environment. Real API results are not guaranteed to be identical across runs.

**Important modeling limitation:** The offline simulation repeatedly samples from one fixed illustrative token distribution. Real autoregressive LLMs update token probabilities after each generated token. The simulator explains sampling mechanics; it does not recreate the full Groq model or its logits.

## Repository layout

```text
llm-sampling-lab/
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .env.example
├── pyproject.toml
├── uv.lock
├── settings.toml
├── README.md
├── docs/
│   └── participant-quick-start.md
├── src/llm_sampling_lab/
│   ├── cli.py
│   ├── config.py
│   ├── experiments.py
│   ├── groq_client.py
│   ├── results.py
│   ├── sampling.py
│   ├── simulator.py
│   └── visualization.py
├── tests/
└── results/               # Generated locally; should be gitignored
```

## Verification and troubleshooting

```shell
uv run pytest -q
```

A complete suite of **93 tests** was reported passing when the Docker workflow was validated (October 2026). Run the tests in your own checkout; the number may change as the project evolves.

| Symptom | First check |
| --- | --- |
| Cannot connect to Docker daemon | Start Docker Desktop; run `docker version` and `docker run --rm hello-world` |
| `no configuration file provided` | Run from the repository root containing `docker-compose.yaml` |
| Groq API key missing | Configure `GROQ_API_KEY` for real experiments; offline simulation does not need it |
| Reports not visible on your computer | Use `docker compose run` with the declared `results` bind mount; inspect `results/` |
| API rate limit/authentication error | Check key permissions and account limits; keep repetitions small |
| CLI argument error | Run `docker compose run --rm sampling-lab --help` |

## Data and security

- Do not commit `.env` or API keys. Keep `.env.example` as a placeholder-only file.
- Treat prompts and model responses in generated JSON reports as potentially sensitive.
- Do not add `results/` to version control unless its contents have been reviewed and deliberately approved for publication.
- Offline experiments do not contact Groq; real experiments do.

## License

No license is asserted here. Add an explicit `LICENSE` file if the project will be redistributed under defined terms.
