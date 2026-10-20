# LLM Sampling Lab

**A hands-on laboratory for understanding LLM sampling strategies through real API experiments and offline simulations.**

LLM Sampling Lab is a Python-based educational project that demonstrates how **Temperature**, **Top-k**, and **Top-p (Nucleus Sampling)** influence probability distributions and text generation.

The project combines three complementary approaches:

1. **Real LLM Experiments:** Explore Temperature and Top-p using the Groq Chat Completions API.
2. **Offline Sampling Simulator:** Apply Temperature, Top-k, and Top-p to a predefined token probability distribution and visualize the results.
3. **Movie Review Generation:** Combine local Temperature and Top-k sampling for review-style selection with real movie review generation using Groq.

The laboratory can run locally with Python and `uv`, or inside Docker containers without requiring students to install Python dependencies.

> **Important:** Groq Chat Completions does not expose a Top-k parameter. In the Movie Review Experiment, Top-k is applied locally to the review-style probability distribution. It does not control Groq's internal token sampling.

---

## 1. Learning Objectives

After completing this laboratory, participants should be able to:

- Explain how Temperature changes a probability distribution.
- Describe the difference between Top-k and Top-p sampling.
- Compare theoretical probabilities with observed sampling frequencies.
- Understand how random seeds support reproducible experiments.
- Run real LLM experiments using the Groq API.
- Analyze output diversity and instruction-following behavior.
- Identify potential hallucinations and unsupported information in generated text.
- Save and examine experimental results in JSON format.
- Generate PNG charts of sampling distributions.
- Run Python-based experiments using Docker and Docker Compose.

---

## 2. Project Features

| Feature                      | Description                                                              |
| ---------------------------- | ------------------------------------------------------------------------ |
| Temperature Experiment       | Compare text generation at different temperature values                  |
| Top-p Experiment             | Explore nucleus sampling through real Groq API requests                  |
| Local Temperature Simulation | Observe probability scaling without API requests                         |
| Local Top-k Simulation       | Restrict token candidates to the highest-probability tokens              |
| Local Top-p Simulation       | Select candidates using cumulative probability thresholds                |
| Random Token Sampling        | Compare expected probabilities with observed frequencies                 |
| Movie Review Experiment      | Generate Turkish film reviews using local style selection and Groq       |
| JSON Persistence             | Save experimental results incrementally                                  |
| Data Visualization           | Generate PNG charts using Matplotlib                                     |
| Automated Testing            | Validate sampling algorithms, persistence, CLI behavior, and integration |
| Docker Support               | Run the project without installing Python locally                        |

---

## 3. Technology Stack

| Component               | Technology           |
| ----------------------- | -------------------- |
| Programming Language    | Python 3.12          |
| Dependency Management   | uv                   |
| LLM Provider            | Groq                 |
| Default Model           | `openai/gpt-oss-20b` |
| Numerical Computing     | NumPy                |
| Visualization           | Matplotlib           |
| Configuration           | TOML                 |
| Testing                 | pytest               |
| Containerization        | Docker               |
| Container Orchestration | Docker Compose       |
| Result Storage          | JSON                 |

---

## 4. Getting Started

### 4.1. Prerequisites

For the recommended Docker workflow, you need:

- Git
- Docker Desktop or Docker Engine with Docker Compose
- An internet connection for downloading the project and building the Docker image
- A Groq API key for real LLM experiments only

**You do not need to install Python, uv, or Python libraries when using Docker.**

### 4.2. Clone the Repository

```bash
git clone https://github.com/Burakkylmz/llm-sampling-lab.git

cd llm-sampling-lab
```

If the repository is private, your GitHub account must have permission to access it.

### 4.3. Build the Docker Image

```bash
docker compose build
```

Docker automatically prepares the Python runtime and installs the required dependencies.

### 4.4. Check the CLI

```bash
docker compose run --rm sampling-lab --help
```

This command displays the available experiments and simulation options.

---

## 5. Offline Sampling Experiments

Offline simulations do not require a Groq API key.

The simulator uses the following educational token distribution:

| Token   | Initial Probability |
| ------- | ------------------: |
| AI      |                0.40 |
| Machine |                0.25 |
| Deep    |                0.15 |
| Neural  |                0.12 |
| Data    |                0.08 |

Each simulation applies a sampling transformation and then performs random token selection.

### 5.1. Temperature Simulation

```bash
docker compose run --rm sampling-lab --simulate temperature --samples 1000 --seed 42
```

Configured Temperature values:

- 0.5
- 1.0
- 1.5

**Expected observations:**

- Temperature below 1 produces a sharper distribution.
- Temperature equal to 1 preserves the original distribution.
- Temperature above 1 produces a flatter distribution.

### 5.2. Top-k Simulation

```bash
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
```

Configured Top-k values:

- 1
- 3
- 5

**Expected observations:**

- Top-k = 1 retains only the most probable token.
- Top-k = 3 retains the three most probable tokens.
- Top-k = 5 retains all five tokens.

After filtering, the remaining token probabilities are normalized.

### 5.3. Top-p Simulation

```bash
docker compose run --rm sampling-lab --simulate top-p --samples 1000 --seed 42
```

Configured Top-p values:

- 0.3
- 0.7
- 0.9

**Expected observations:**

Top-p selects the smallest group of high-probability tokens whose cumulative probability reaches the specified threshold.

Unlike Top-k, the number of retained tokens is not fixed.

### 5.4. Expected vs. Observed Probabilities

Every offline simulation compares two distributions:

- **Expected:** The theoretical probability after applying the selected sampling strategy.
- **Observed:** The relative frequency of token selections during the simulation.

The observed frequencies generally approach theoretical probabilities as the sample count increases, but they do not necessarily match exactly.

The `--seed` parameter helps make experiments reproducible.

PNG charts are generated automatically under:

`results/visualizations/`

---

## 6. Groq API Configuration

Real LLM experiments require a Groq API key.

You can obtain your own key from:

https://console.groq.com/keys

### 6.1. Create the Environment File

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your own API key:

```dotenv
GROQ_API_KEY=your_own_groq_api_key_here
```

**Never commit `.env` or share your API key.**

### 6.2. Verify API Connectivity

```bash
docker compose run --rm sampling-lab --check-api
```

A successful connection should produce a response from the configured Groq model.

---

## 7. Real LLM Sampling Experiments

The project supports two real API sampling experiments.

These experiments use a shared prompt that asks the model to generate a short advertising slogan for an electric car brand.

### 7.1. Temperature Experiment

```bash
docker compose run --rm sampling-lab --experiment temperature --repetitions 3
```

Configured values:

| Parameter   | Value                   |
| ----------- | ----------------------- |
| Temperature | 0.1, 0.5, 1.0           |
| Top-p       | 1.0                     |
| Repetitions | 3 per temperature value |

The experiment changes Temperature while keeping Top-p fixed.

Use the generated responses to examine variation across repeated requests.

### 7.2. Top-p Experiment

```bash
docker compose run --rm sampling-lab --experiment top-p --repetitions 3
```

Configured values:

| Parameter   | Value             |
| ----------- | ----------------- |
| Top-p       | 0.3, 0.7, 1.0     |
| Temperature | 0.7               |
| Repetitions | 3 per Top-p value |

The experiment changes Top-p while keeping Temperature fixed.

### 7.3. Unique Response Ratio

For the Temperature and Top-p experiments, the project calculates:

**Unique Response Ratio (URR) = Number of Unique Responses / Total Responses**

Example:

If the model produces five responses and three are different:

URR = 3 / 5 = 0.60

**Interpretation:**

- A lower ratio indicates more repeated exact responses.
- A higher ratio indicates greater exact-text variation.
- URR does not measure semantic diversity, factual accuracy, or response quality.
- With only one repetition, URR is always 1.0.

Generated reports are saved under:

- `results/temperature/`
- `results/top_p/`

---

## 8. Movie Review Generation Experiment

The Movie Review Experiment demonstrates how local sampling and real LLM generation can be combined.

It uses a fictional science-fiction drama titled **Son İstasyon**.

### 8.1. Experiment Workflow

The experiment follows these steps:

1. Apply Temperature scaling to a local review-style probability distribution.
2. Apply local Top-k filtering.
3. Randomly select a review style using a predefined seed.
4. Construct a movie review prompt.
5. Generate a Turkish movie review using the Groq API.
6. Validate basic structural requirements.
7. Persist the result in JSON format.

### 8.2. Review Styles

The local style distribution contains four candidates:

| Review Style | Initial Probability |
| ------------ | ------------------: |
| Positive     |                0.40 |
| Critical     |                0.30 |
| Technical    |                0.20 |
| Emotional    |                0.10 |

Temperature and Top-k are applied locally to this distribution before a review style is selected.

### 8.3. Experiment Scenarios

| Scenario | Temperature | Local Top-k | Groq Top-p |
| -------- | ----------: | ----------: | ---------: |
| 1        |         0.3 |           1 |        1.0 |
| 2        |         0.3 |           3 |        1.0 |
| 3        |         1.0 |           1 |        1.0 |
| 4        |         1.0 |           3 |        1.0 |

The same Temperature value is used for local style selection and Groq text generation.

**Important limitation:** Local Top-k filters the possible review styles. It does not modify the token probabilities inside the Groq model.

### 8.4. Run the Experiment

```bash
docker compose run --rm sampling-lab --experiment movie-reviews
```

The command generates four movie reviews.

The experiment does not use the `--repetitions` option.

### 8.5. Output Validation

The experiment performs deterministic checks for:

- Presence of the specified movie title
- Required review length of 80–100 words

Validation warnings are stored with each result.

**A completed experiment may still contain validation warnings.**

The current validator does not reliably detect invented story details, semantic inconsistencies, grammatical problems, or review-style violations.

### 8.6. Movie Review JSON Reports

Reports are saved under:

`results/movie_reviews/`

Each report contains:

- Experiment ID
- Model
- Movie information
- Scenario parameters
- Local review style
- Generated review
- Validation issues
- Experiment status
- UTC timestamps

Results are saved incrementally after each successful response.

---

## 9. Results and Reports

The project generates two main output formats.

| Format | Purpose                                  |
| ------ | ---------------------------------------- |
| JSON   | Store real LLM experiment data           |
| PNG    | Visualize offline sampling distributions |

Generated output structure:

```text
results/
├── temperature/
│   └── <experiment>.json
├── top_p/
│   └── <experiment>.json
├── movie_reviews/
│   └── <experiment>.json
└── visualizations/
    ├── temperature_0p5.png
    ├── top-k_3.png
    └── top-p_0p7.png
```

These filenames are examples; actual files depend on the experiments executed.

### 9.1. JSON Persistence Behavior

Real experiments use incremental persistence.

After each successful API response, the JSON report is updated.

The report can have the following statuses:

| Status      | Meaning                                 |
| ----------- | --------------------------------------- |
| `running`   | The experiment has started              |
| `completed` | All planned scenarios finished          |
| `failed`    | An exception interrupted the experiment |

When a handled API error occurs, previously saved responses remain available.

An unexpected process termination may leave the report in the `running` state.

### 9.2. Docker Volume Mounting

Docker Compose mounts the host `results/` directory into the container.

As a result, generated JSON reports and PNG charts remain available on your computer even after the container is removed.

---

## 10. Local Development with Python and uv

Docker is recommended for participants.

Developers can also run the project directly using Python.

### 10.1. Requirements

- Python 3.12
- uv

### 10.2. Install Dependencies

```bash
uv sync
```

### 10.3. Run Offline Simulations

```bash
uv run sampling-lab --simulate temperature

uv run sampling-lab --simulate top-k

uv run sampling-lab --simulate top-p
```

### 10.4. Run Real Groq Experiments

Before running real API experiments, configure `GROQ_API_KEY` in your terminal environment.

For direct `uv run` commands, a `.env` file used by Docker Compose is not automatically loaded by the application.

```bash
uv run sampling-lab --experiment temperature --repetitions 3

uv run sampling-lab --experiment top-p --repetitions 3

uv run sampling-lab --experiment movie-reviews
```

---

## 11. CLI Reference

| Command / Option             | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `--help`                     | Display CLI options                                      |
| `--check-api`                | Verify Groq API connectivity                             |
| `--experiment temperature`   | Run real Temperature experiments                         |
| `--experiment top-p`         | Run real Top-p experiments                               |
| `--experiment movie-reviews` | Generate four movie reviews                              |
| `--repetitions N`            | Set repetitions for Temperature or Top-p API experiments |
| `--simulate temperature`     | Run offline Temperature simulation                       |
| `--simulate top-k`           | Run offline Top-k simulation                             |
| `--simulate top-p`           | Run offline Top-p simulation                             |
| `--samples N`                | Set offline sample count                                 |
| `--seed N`                   | Set offline random seed                                  |

`--check-api`, `--experiment`, and `--simulate` are mutually exclusive.

`--samples` and `--seed` apply only to offline simulations.

`--repetitions` applies only to Temperature and Top-p API experiments.

---

## 12. Configuration

The project uses `settings.toml` for its main settings.

Example:

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

These settings control the original Groq sampling experiments and offline simulations.

The Movie Review Experiment currently uses predefined scenarios in `movie_reviews.py` and a fixed seed in its CLI runner.

Changing `settings.toml` does not change the Movie Review scenarios.

When using Docker Compose, `settings.toml` is mounted read-only into the container. Changes made to this file on the host are available to subsequent Compose runs.

---

## 13. Project Structure

```text
llm-sampling-lab/
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── docker-compose.yaml
├── README.md
├── pyproject.toml
├── settings.toml
├── uv.lock
│
├── docs/
│   └── participant-quick-start.md
│
├── src/
│   └── llm_sampling_lab/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── experiments.py
│       ├── groq_client.py
│       ├── movie_reviews.py
│       ├── results.py
│       ├── sampling.py
│       ├── simulator.py
│       └── visualization.py
│
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_experiments.py
│   ├── test_groq_client.py
│   ├── test_movie_persistence.py
│   ├── test_movie_reviews.py
│   ├── test_persistence_integration.py
│   ├── test_phase5_e2e.py
│   ├── test_results.py
│   ├── test_sampling.py
│   ├── test_simulator.py
│   └── test_visualization.py
│
└── results/
    ├── temperature/
    ├── top_p/
    ├── movie_reviews/
    └── visualizations/
```

The `results/` directory is generated locally and is excluded from version control.

---

## 14. Automated Testing

The project includes automated tests for:

- Configuration loading and validation
- Groq client behavior
- Temperature scaling
- Top-k filtering
- Top-p filtering
- Random token sampling
- Reproducible experiments
- JSON persistence
- API failure recovery
- Movie review generation
- CLI integration
- Docker-compatible offline workflows
- PNG visualization

### Run All Tests

```bash
uv run pytest -q
```

### Run Movie Review Tests

```bash
uv run pytest tests/test_movie_reviews.py -v

uv run pytest tests/test_movie_persistence.py -v
```

Tests use mocks where appropriate, allowing the test suite to run without making real Groq API requests.

---

## 15. Troubleshooting

| Problem                                  | Recommended Action                                |
| ---------------------------------------- | ------------------------------------------------- |
| Docker Engine is not running             | Start Docker Desktop and verify `docker version`  |
| Docker Compose cannot find configuration | Run commands from the repository root             |
| Missing Groq API key                     | Create `.env` and configure `GROQ_API_KEY`        |
| API authentication failed                | Check API key validity                            |
| API rate limit exceeded                  | Reduce repetitions and review account limits      |
| No PNG charts                            | Check `results/visualizations/`                   |
| No JSON report                           | Check experiment output and `results/`            |
| Movie Review validation warnings         | Examine title preservation and word count         |
| Unsupported CLI option                   | Run `docker compose run --rm sampling-lab --help` |

---

## 16. Security and Usage Notes

- Use your own Groq API key.
- Never commit `.env` or real API credentials.
- Do not include secrets in screenshots, shared logs, or issues.
- Keep generated reports out of version control unless they have been reviewed.
- Real Groq experiments consume API resources and may be subject to usage limits.
- Offline simulations do not send requests to Groq.
- JSON reports may contain prompts and generated content; review them before sharing.
- A random seed does not guarantee identical responses from an external LLM API.

---

## 17. Educational Limitations

This project is intended to explain sampling behavior and support controlled classroom experiments.

The offline simulator uses a fixed educational probability distribution. Real autoregressive LLMs recalculate token probabilities at each generation step.

The Movie Review Experiment uses Top-k for local review-style selection rather than Groq token-level sampling.

The project does not currently measure:

- Input and output token costs
- Cached tokens
- Reasoning token consumption
- Time to First Token (TTFT)
- Total inference latency
- Throughput
- Monetary API costs

These subjects are outside the scope of this laboratory.

---

## 18. Participant Guide

For a step-by-step introduction to Docker setup, offline simulations, real API experiments, and result analysis, see:

[Participant Quick Start](docs/participant-quick-start.md)

---

## 19. License

No explicit open-source license is currently included.

Repository access does not automatically grant permission to redistribute or modify the project beyond applicable permissions and law.
