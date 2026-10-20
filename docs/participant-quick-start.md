# Participant Quick Start — LLM Sampling Lab

This guide is for learners. It explains how to run the laboratory with **Docker only**; you do not have to install Python, `uv`, or Python libraries.

## 1. Before class

You need:

1. Docker Desktop running (or Docker Engine plus Docker Compose on Linux).
2. A copy of the `llm-sampling-lab` repository on your computer.
3. Your **own Groq API key only if you will run real model experiments**. Offline simulations require no key.

Open a terminal in the repository root, next to `docker-compose.yaml` and `settings.toml`.

On Windows PowerShell, you can check:

```powershell
docker version
docker compose version
Test-Path .\docker-compose.yaml
```

`docker version` should show both **Client** and **Server** sections. The last command should return `True`.

## 2. Build the Docker image

```powershell
docker compose build
```

The initial build may download base images and Python dependencies. Later runs normally reuse the image.

## 3. Run the offline simulations

No API key is needed for any of these commands.

**Temperature:**

```powershell
docker compose run --rm sampling-lab --simulate temperature --samples 1000 --seed 42
```

**Top-k:**

```powershell
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
```

**Top-p:**

```powershell
docker compose run --rm sampling-lab --simulate top-p --samples 1000 --seed 42
```

Each command reads its parameter values from `settings.toml`, prints **Expected** and **Observed** probabilities, and creates PNG charts under `results/visualizations/` on your computer.

On Windows, open the chart directory with:

```powershell
Invoke-Item .\results\visualizations
```

### What should you observe?

- **Temperature below 1:** greater weight on more likely tokens.
- **Temperature above 1:** flatter probabilities across tokens.
- **Top-k = 1:** only one candidate survives.
- **Top-k = 3:** only the three most likely tokens can be selected.
- **Top-p:** the number of candidates changes with the cumulative probability threshold.
- **Expected vs. Observed:** frequencies tend to approach probabilities as the number of samples grows, but they need not match exactly.

## 4. Optional: Run a real Groq API experiment

**This section sends live requests to Groq.** Use your own key and review the provider's usage and quota conditions. Never share the key with other participants.

In the repository root, create a `.env` file. If `.env.example` exists, Windows users can copy it:

```powershell
Copy-Item .env.example .env
```

Open `.env` in a text editor, then replace the placeholder:

```dotenv
GROQ_API_KEY=your_own_groq_api_key_here
```

Do not put quotes or extra spaces around the key unless required by your shell configuration.

Test connectivity:

```powershell
docker compose run --rm sampling-lab --check-api
```

Run a short **temperature experiment**:

```powershell
docker compose run --rm sampling-lab --experiment temperature --repetitions 1
```

Run a short **top-p experiment**:

```powershell
docker compose run --rm sampling-lab --experiment top-p --repetitions 1
```

The results will appear under `results/temperature/` and `results/top_p/` as JSON files. The console prints the final report path.

> **Interpretation caution:** One repetition produces a Unique Response Ratio of 1.0 by definition. Use more repetitions to explore exact-text diversity, and do not confuse exact-text diversity with semantic or creative quality.

## 5. Where do the files go?

```text
results/
├── visualizations/   # PNG charts from offline simulations
├── temperature/      # JSON reports from real temperature experiments
└── top_p/            # JSON reports from real top-p experiments
```

`docker compose` binds this directory to the container. This is why files persist when you use `--rm` to delete a completed container.

To list recent JSON reports on Windows:

```powershell
Get-ChildItem .\results\temperature\*.json
Get-ChildItem .\results\top_p\*.json
```

You can open a JSON file in VS Code or another text editor. Avoid publishing it without checking its prompt and generated content.

## 6. Change experiment settings

The file `settings.toml` controls the active Groq model and parameter lists.

Relevant offline settings:

```toml
[local_sampling]
top_k_values = [1, 3, 5]
top_p_values = [0.3, 0.7, 0.9]
temperature_values = [0.5, 1.0, 1.5]
```

Compose mounts this configuration read-only inside the container, but you can edit the file on your own computer. Keep parameter values within the ranges validated by the application.

## 7. Troubleshooting

| Message or issue | Solution |
| --- | --- |
| `failed to connect to the docker API` | Open Docker Desktop, then run `docker version`. Docker Desktop CLI users can try `docker desktop start`. |
| `no configuration file provided: not found` | Switch into the repository root and ensure `docker-compose.yaml` exists. |
| `GROQ_API_KEY ... missing or empty` | Create/update `.env`; verify the key is yours and non-empty. This is only needed for live API commands. |
| No PNG/JSON files on host | Use the Compose commands from this guide; check the `./results:/app/results` volume in `docker-compose.yaml`. |
| API authentication or quota error | Verify your Groq account and usage limits, and try fewer repetitions. |
| Unsupported CLI option | Run `docker compose run --rm sampling-lab --help`. |

## 8. After class

Keep the generated charts and JSON reports for analysis. Do **not** upload your `.env` file or expose `GROQ_API_KEY` in screenshots, chat, GitHub issues, or shared documents.

For project architecture, the complete command reference, and local development instructions, see the repository [README](../README.md).
