# Participant Quick Start — LLM Sampling Lab

**A practical guide to running Temperature, Top-k, and Top-p experiments using Docker and the Groq API.**

This guide explains how to set up and run the LLM Sampling Lab on your own computer.

You will perform three types of activities:

1. **Offline Sampling Simulations:** Explore Temperature, Top-k, and Top-p without calling an LLM API.
2. **Real Groq API Experiments:** Observe how Temperature and Top-p affect generated advertising slogans.
3. **Movie Review Generation:** Combine local Temperature and Top-k sampling with real LLM-generated movie reviews.

The recommended workflow uses **Docker Compose**.

You do not need to install Python, `uv`, NumPy, Matplotlib, or other Python libraries on your computer.

---

## 1. Before Class

### 1.1. Required Software

You need:

- Docker Desktop, or Docker Engine with Docker Compose support.
- Git, if you plan to clone the repository.
- A terminal such as Windows PowerShell.
- An internet connection for downloading the project, building the image, and running real API experiments.
- A text editor, preferably Visual Studio Code.

A **Groq API key is optional** for offline simulations but required for real LLM experiments.

### 1.2. Verify Docker

Open PowerShell and run:

```powershell
docker version
```

The output should contain both **Client** and **Server** information.

Then check Docker Compose:

```powershell
docker compose version
```

If Docker is not running, start Docker Desktop and repeat these commands.

### 1.3. Obtain the Project

The project is available on GitHub:

https://github.com/Burakkylmz/llm-sampling-lab

Clone it using Git:

```powershell
git clone https://github.com/Burakkylmz/llm-sampling-lab.git
```

Navigate into the project directory:

```powershell
cd llm-sampling-lab
```

Alternatively, download the project using **GitHub → Code → Download ZIP**, then extract the archive and open the extracted project directory.

> **Repository access:** If the repository is private, your GitHub account must have permission to access it.

### 1.4. Verify the Project Directory

The project directory should contain:

```text
llm-sampling-lab/
├── Dockerfile
├── docker-compose.yaml
├── settings.toml
├── pyproject.toml
├── README.md
├── docs/
├── src/
└── tests/
```

Run:

```powershell
Test-Path .\docker-compose.yaml
```

Expected output:

```text
True
```

Always execute the Docker Compose commands from the project root.

---

## 2. Build the Docker Image

Before running the experiments, build the Docker image:

```powershell
docker compose build
```

Docker will:

1. Download the required base images.
2. Prepare a Python 3.12 runtime.
3. Install the project dependencies.
4. Package the application into a Docker image.

The first build requires internet access and may download several dependencies.

After the build is complete, verify that the application is available:

```powershell
docker compose run --rm sampling-lab --help
```

The CLI should display the supported experiments and simulation commands.

### Why Use Docker?

Docker provides a consistent execution environment.

The same laboratory can run on different computers without requiring participants to install and configure the Python dependencies manually.

When application source files change, rebuild the Docker image:

```powershell
docker compose build
```

---

## 3. Offline Sampling Simulations

Offline simulations demonstrate sampling algorithms using a predefined probability distribution.

**These simulations do not require a Groq API key.**

### 3.1. Educational Token Distribution

The simulator uses five tokens:

| Token     | Initial Probability |
| --------- | ------------------: |
| AI        |                0.40 |
| Machine   |                0.25 |
| Deep      |                0.15 |
| Neural    |                0.12 |
| Data      |                0.08 |
| **Total** |            **1.00** |

Each simulation transforms this distribution using a sampling strategy.

The application then samples tokens repeatedly and compares theoretical probabilities with observed frequencies.

### 3.2. Temperature Simulation

Run:

```powershell
docker compose run --rm sampling-lab --simulate temperature --samples 1000 --seed 42
```

The default configuration uses:

| Temperature | Expected Behavior     |
| ----------- | --------------------- |
| 0.5         | Sharper distribution  |
| 1.0         | Original distribution |
| 1.5         | Flatter distribution  |

**What to observe**

When Temperature is below 1, higher-probability tokens receive relatively greater weight.

When Temperature is above 1, the distribution becomes flatter, giving lower-probability tokens relatively more weight.

Temperature changes the probability distribution; it does not directly specify how many tokens remain eligible for selection.

### 3.3. Top-k Simulation

Run:

```powershell
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
```

The default Top-k values are:

| Top-k | Expected Behavior                                |
| ----- | ------------------------------------------------ |
| 1     | Only the highest-probability token remains       |
| 3     | Only the three highest-probability tokens remain |
| 5     | All five tokens remain                           |

After filtering, the probabilities of the selected tokens are normalized.

**What to observe**

With Top-k = 1, only the token `AI` can be selected.

With Top-k = 3, only `AI`, `Machine`, and `Deep` can be selected.

With Top-k = 5, the original candidate set is preserved.

### 3.4. Top-p Simulation

Run:

```powershell
docker compose run --rm sampling-lab --simulate top-p --samples 1000 --seed 42
```

The default values are:

- 0.3
- 0.7
- 0.9

Top-p is also called **Nucleus Sampling**.

It selects the smallest group of tokens, ordered by descending probability, whose cumulative probability reaches the chosen threshold.

**What to observe**

Unlike Top-k, Top-p does not keep a fixed number of candidates.

The number of remaining tokens depends on the probability distribution and the selected threshold.

### 3.5. Expected vs. Observed Probabilities

Each simulation prints:

- **Expected Probability:** The theoretical probability after the sampling transformation.
- **Observed Probability:** The relative frequency produced by random sampling.

For example, a token with a theoretical probability of 0.40 might have an observed frequency of 0.39 or 0.42 after 1,000 samples.

This variation is normal.

Try increasing the sample count:

```powershell
docker compose run --rm sampling-lab --simulate temperature --samples 10000 --seed 42
```

Compare the results with the 1,000-sample experiment.

Larger samples generally provide a closer approximation of the theoretical distribution.

### 3.6. Open the Generated Charts

The application creates PNG charts under:

```text
results/visualizations/
```

On Windows PowerShell, open the directory:

```powershell
Invoke-Item .\results\visualizations
```

Each chart compares expected and observed token probabilities.

### 3.7. Offline Experiment Questions

After running the simulations, consider:

1. Why does Temperature = 0.5 make the probability distribution sharper?
2. Why does Top-k = 1 always select the same token?
3. What is the main difference between Top-k and Top-p?
4. Why do observed probabilities differ slightly from theoretical probabilities?
5. How does increasing the sample count affect the observed distribution?

---

## 4. Configure the Groq API

The remaining experiments use a real LLM through Groq.

**Real API experiments require your own Groq API key.**

They also consume API resources and may be subject to account limits.

### 4.1. Obtain an API Key

Visit:

https://console.groq.com/keys

Sign in and create an API key for your account.

Never share this key with other participants.

### 4.2. Create the Environment File

From the project root, run:

```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code or another text editor.

Replace the placeholder with your actual Groq API key:

```dotenv
GROQ_API_KEY=your_own_groq_api_key_here
```

Save the file.

**Security requirements**

- Never commit `.env` to GitHub.
- Never share your API key in screenshots.
- Do not paste your API key into public messages or shared documents.
- Do not replace `.env.example` with a real API key.

### 4.3. Verify Groq Connectivity

Run:

```powershell
docker compose run --rm sampling-lab --check-api
```

A successful result confirms that the application can communicate with Groq using the configured model.

The default model is:

```text
openai/gpt-oss-20b
```

If authentication fails, verify the API key and your account permissions.

---

## 5. Real LLM Sampling Experiments

The project includes two controlled experiments that use real Groq API requests.

Both experiments ask the model to generate a short advertising slogan for an electric car brand.

The purpose is to observe how different sampling settings influence generated text.

### 5.1. Temperature Experiment

Run:

```powershell
docker compose run --rm sampling-lab --experiment temperature --repetitions 3
```

The default configuration uses:

| Parameter   | Value                   |
| ----------- | ----------------------- |
| Temperature | 0.1, 0.5, 1.0           |
| Top-p       | 1.0                     |
| Repetitions | 3 per Temperature value |

In this experiment, Temperature changes while Top-p remains fixed.

The command requests three responses for each Temperature value, resulting in nine planned API requests.

**What to observe**

Compare the generated slogans:

- Do lower Temperature values produce similar wording?
- Does a higher Temperature increase text variation?
- Are all responses relevant to the advertising task?
- Does more variation necessarily mean higher quality?

### 5.2. Top-p Experiment

Run:

```powershell
docker compose run --rm sampling-lab --experiment top-p --repetitions 3
```

The default configuration uses:

| Parameter   | Value             |
| ----------- | ----------------- |
| Top-p       | 0.3, 0.7, 1.0     |
| Temperature | 0.7               |
| Repetitions | 3 per Top-p value |

In this experiment, Top-p changes while Temperature remains fixed.

**What to observe**

Compare the generated slogans across different Top-p settings.

A smaller Top-p value limits token selection to a narrower cumulative probability mass.

A larger Top-p value permits a broader range of candidate tokens.

This does not guarantee that every output will be different.

### 5.3. Understanding Unique Response Ratio

The application calculates a metric called **Unique Response Ratio (URR)**.

Formula:

```text
Unique Response Ratio = Unique Responses / Total Responses
```

Example:

If the application generates five responses and three are different:

```text
URR = 3 / 5 = 0.60
```

**Important limitations**

URR compares exact response strings.

Two texts with the same meaning but different wording are counted as different.

URR does not measure:

- Factual accuracy
- Semantic diversity
- Creativity
- Readability
- Overall response quality

With only one response, URR equals 1.0 automatically.

### 5.4. Locate the JSON Reports

Temperature experiment results:

```text
results/temperature/
```

Top-p experiment results:

```text
results/top_p/
```

Each report contains the experiment parameters, generated responses, execution status, and summary metrics.

---

## 6. Movie Review Generation Experiment

The Movie Review Experiment combines **local probabilistic sampling** with real LLM text generation.

This experiment is different from the previous Temperature and Top-p experiments.

### 6.1. Experiment Goal

The application generates Turkish movie reviews for a fictional science-fiction drama.

Movie information:

| Field | Value                                                                                                                                              |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Title | Son İstasyon                                                                                                                                       |
| Genre | Science Fiction / Drama                                                                                                                            |
| Story | A scientist discovers that humanity has only one year to establish a colony on another planet. She must choose between her family and the mission. |

The experiment selects a review style before requesting text from Groq.

### 6.2. Local Review Style Distribution

The application uses four possible review styles:

| Style     | Initial Probability |
| --------- | ------------------: |
| Positive  |                0.40 |
| Critical  |                0.30 |
| Technical |                0.20 |
| Emotional |                0.10 |

Temperature and Top-k are applied to these probabilities locally.

The selected style becomes part of the prompt sent to Groq.

### 6.3. How the Experiment Works

The workflow consists of six steps:

1. **Temperature Scaling:** Transform the probabilities of the four review styles.
2. **Top-k Filtering:** Keep the highest-probability review-style candidates.
3. **Style Selection:** Randomly select one style using the filtered probability distribution.
4. **Prompt Construction:** Add the movie information and selected style to the instructions.
5. **Groq Generation:** Request a Turkish movie review.
6. **Validation and Persistence:** Check structural requirements and save the result in JSON.

> **Important technical distinction:** Groq Chat Completions does not expose Top-k as a request parameter. In this experiment, Top-k is applied locally to review-style selection, not to the model's internal token sampling.

### 6.4. Experiment Scenarios

The application uses four predefined scenarios:

| Scenario | Temperature | Local Top-k | Groq Top-p |
| -------- | ----------: | ----------: | ---------: |
| 1        |         0.3 |           1 |        1.0 |
| 2        |         0.3 |           3 |        1.0 |
| 3        |         1.0 |           1 |        1.0 |
| 4        |         1.0 |           3 |        1.0 |

The same Temperature value is applied both to local review-style sampling and Groq text generation.

Therefore, differences between results should not automatically be attributed to only one parameter.

### 6.5. Run the Movie Review Experiment

Execute:

```powershell
docker compose run --rm sampling-lab --experiment movie-reviews
```

The application sends four real requests to Groq.

For each scenario, the terminal displays:

- Scenario number
- Temperature
- Local Top-k
- Selected review style
- Generated movie review
- Validation warnings, if any

The Movie Review Experiment uses four predefined scenarios and does not support the `--repetitions` option.

### 6.6. Understanding the Results

Consider the following observations:

**Top-k = 1**

Only the most probable review style remains eligible.

With the current probability distribution, this means `Positive`.

**Top-k = 3**

Three review styles remain eligible.

However, the application may still select `Positive`.

Increasing Top-k does not guarantee that a different style will be selected.

**Temperature = 0.3**

The local style distribution becomes more concentrated.

Groq also receives a lower Temperature for text generation.

**Temperature = 1.0**

The original local style probabilities are preserved before Top-k filtering.

Groq receives Temperature = 1.0 for text generation.

### 6.7. Output Validation

The application checks two basic requirements:

1. Does the generated review contain the specified movie title?
2. Does the review contain between 80 and 100 words?

Possible validation warnings:

```text
movie_title_missing_or_changed
word_count_out_of_range
```

These warnings are included in the JSON report.

**A completed experiment can contain validation warnings.**

For example, a Groq API request can succeed even when the model produces a review containing fewer than 80 words.

The current validator does not reliably detect:

- Invented characters
- Unsupported story details
- Semantic inconsistencies
- Grammatical errors
- Failure to follow the selected review style

These issues require additional evaluation.

### 6.8. JSON Output

The report is saved under:

```text
results/movie_reviews/
```

The report includes:

| Field               | Meaning                                        |
| ------------------- | ---------------------------------------------- |
| `experiment_id`     | Unique experiment identifier                   |
| `experiment_type`   | Experiment category                            |
| `model`             | Groq model identifier                          |
| `movie`             | Movie information                              |
| `parameters`        | Sampling parameters and scenarios              |
| `status`            | Experiment execution status                    |
| `results`           | Generated reviews and their metadata           |
| `validation_issues` | Structural validation warnings for each result |
| `summary`           | Total and completed scenario counts            |

The report also contains UTC timestamps.

### 6.9. Inspect the JSON Report in PowerShell

Find the most recent report:

```powershell
$report = Get-ChildItem ".\results\movie_reviews\*.json" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
```

Load the JSON:

```powershell
$data = Get-Content $report.FullName -Raw |
    ConvertFrom-Json
```

Display the experiment summary:

```powershell
Write-Host "Experiment: $($data.experiment_type)"
Write-Host "Status: $($data.status)"
Write-Host "Completed Scenarios: $($data.summary.completed_scenarios)"
Write-Host "Validation Warnings: $($data.summary.validation_warning_count)"
```

Compare the scenarios:

```powershell
$data.results |
    Select-Object scenario, temperature, local_top_k, review_style |
    Format-Table -AutoSize
```

The expected completed scenario count is four if all requests succeed.

The number of validation warnings may vary.

### 6.10. Movie Review Discussion Questions

After completing the experiment, consider:

1. Why can Top-k = 3 still select the `Positive` review style?
2. What happens to the local style distribution when Temperature changes?
3. Can two identical sampling configurations produce different reviews?
4. Does higher Temperature guarantee better creativity?
5. Can the model follow the review-style instruction but violate the word-count requirement?
6. What is the difference between a successful API request and a valid generated response?
7. Why is local Top-k different from token-level Top-k inside an LLM?
8. Why are four generated reviews insufficient to make strong statistical claims about Temperature?

---

## 7. Understanding the Results Directory

All generated files are saved in the project's `results/` directory.

```text
results/
├── temperature/       # Real Temperature experiment JSON reports
├── top_p/             # Real Top-p experiment JSON reports
├── movie_reviews/     # Movie Review experiment JSON reports
└── visualizations/    # Offline sampling PNG charts
```

Docker Compose mounts the project's `results/` directory into the container.

This means generated reports remain on your computer after the container is removed.

### List the JSON Reports

```powershell
Get-ChildItem .\results\temperature\*.json
```

```powershell
Get-ChildItem .\results\top_p\*.json
```

```powershell
Get-ChildItem .\results\movie_reviews\*.json
```

### Open a Result Directory

```powershell
Invoke-Item .\results
```

You can inspect JSON reports using VS Code or another text editor.

---

## 8. Experiment Configuration

The application uses `settings.toml` for its main configuration.

The default settings include:

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

### Changing the Settings

You may edit `settings.toml` to change the original Temperature and Top-p experiments or the offline simulations.

For example, changing:

```toml
[experiments]
repetitions = 2
```

changes the default repetitions for the two original real API sampling experiments.

The CLI option `--repetitions` can override this value.

**Movie Review configuration:** The Movie Review Experiment currently uses predefined scenarios in `movie_reviews.py` and a fixed seed in the CLI runner. It does not read its scenario list from `settings.toml`.

### Important Configuration Note

Docker Compose mounts `settings.toml` read-only inside the container.

Changes made to this file on your computer are available to subsequent Compose runs.

If you modify Python source code, rebuild the Docker image.

---

## 9. Automated Tests

The repository contains automated tests for sampling functions, configuration, Groq integration, CLI behavior, and JSON persistence.

Participants using the Docker workflow do not need to install a local Python environment.

For developers using Python and `uv`, run:

```powershell
uv run pytest -q
```

The automated tests use mocks where appropriate and do not require actual Groq API calls.

For further development instructions, see the project README.

---

## 10. Troubleshooting

| Problem                                    | Recommended Solution                                            |
| ------------------------------------------ | --------------------------------------------------------------- |
| Docker daemon is unavailable               | Start Docker Desktop and run `docker version`                   |
| `no configuration file provided`           | Open the repository root containing `docker-compose.yaml`       |
| Docker image is outdated                   | Run `docker compose build`                                      |
| `GROQ_API_KEY` is missing                  | Create `.env` and enter your own API key                        |
| Groq authentication failed                 | Check whether your API key is valid                             |
| Groq rate limit exceeded                   | Reduce experiment repetitions or review account limits          |
| No PNG charts                              | Check `results/visualizations/`                                 |
| No JSON report                             | Check the terminal output and the relevant `results/` directory |
| Movie Review returns validation warnings   | Inspect the review for title or word-count violations           |
| Unsupported experiment option              | Run `docker compose run --rm sampling-lab --help`               |
| GitHub clone is denied                     | Verify repository access permissions                            |
| Source changes are not reflected in Docker | Rebuild the image before rerunning the experiment               |

---

## 11. Important Technical Limitations

The laboratory is designed for educational experiments.

Keep the following limitations in mind.

**1. Offline simulation is not a complete LLM.**

The offline simulator works with a fixed token probability distribution.

Real autoregressive language models calculate a new token probability distribution during generation.

**2. Top-k is not directly configurable through the Groq API used in this project.**

The Movie Review Experiment applies Top-k locally to review-style selection.

**3. A random seed does not guarantee identical Groq responses.**

Seeds make local sampling reproducible under the same conditions, but real API generation may still vary.

**4. Higher diversity does not automatically mean better output quality.**

Sampling affects output selection, but factual correctness, clarity, and instruction following are separate concerns.

**5. Structural validation is limited.**

The Movie Review validator checks only the presence of the movie title and the word-count range.

**6. This laboratory is not a performance benchmark.**

It does not currently evaluate inference latency, token usage, throughput, or API cost.

---

## 12. After Class

Keep your generated JSON reports and PNG charts for further analysis.

Review the differences between:

- Temperature = 0.5 and Temperature = 1.5 in the offline simulator.
- Top-k = 1 and Top-k = 3.
- Top-p = 0.3 and Top-p = 0.9.
- Low and high Temperature settings in real Groq experiments.
- Positive and Critical review styles in the Movie Review Experiment.
- Successful API execution and successful instruction following.

Do not upload `.env` or expose your Groq API key in GitHub issues, screenshots, chat messages, or shared documents.

For complete project documentation, technology details, architecture, and development instructions, see:

[LLM Sampling Lab — README](../README.md)

---

**End of Participant Quick Start**