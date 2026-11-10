# Run LLM Sampling Lab Web UI in Docker

This guide launches the **existing Streamlit app** from the repository with Docker Compose.
The legacy CLI still uses the same image and can be executed on demand.

## Requirements

- Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin (Linux)
- Repository files, including `settings.toml`, `uv.lock`, and `streamlit_app.py`
- Internet connection to build the image for the first time
- **No Python, uv, or Streamlit installation on the host**

## Quick start (Windows PowerShell)

Run these commands in the repository root:

```powershell
docker compose up --build -d
docker compose ps
```

Open **http://localhost:8501** in your browser.

The `ui` service listens on port 8501 inside the container and maps it to
`127.0.0.1:8501` on your host. The binding is deliberately local-only; it does
not expose the lab to other computers on the network by default.

Useful commands:

```powershell
docker compose logs -f ui
docker compose restart ui
docker compose down
```

`docker compose down` stops and removes containers, not the host's report
files. Existing CLI workflows still work:

```powershell
docker compose run --rm sampling-lab --help
docker compose run --rm sampling-lab --simulate top-k --samples 1000 --seed 42
```

The CLI service uses the `cli` profile to prevent it from starting when only
the dashboard is requested; explicit `docker compose run` works as before.

## Provider keys (optional for initial launch)

Create a private `.env` file if you want to perform live API experiments:

```powershell
Copy-Item .env.example .env
```

Edit the file and set your own keys:

```dotenv
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=
STREAMLIT_PORT=8501
```

- `GROQ_API_KEY` is needed to execute Groq requests.
- `OPENAI_API_KEY` is needed to execute OpenAI requests.
- Without API keys, the dashboard can still start and historical reports can
  be inspected. Local Ollama requires an accessible Ollama server.
- Do not put secrets in `Dockerfile`, `docker-compose.yaml`, or tracked
  `settings.toml`. The existing `.gitignore` and `.dockerignore`
  exclude `.env`.
- `docker compose up --build -d` rebuilds after code updates. If you only
  edit keys in `.env`, run `docker compose up -d --force-recreate ui` to
  recreate the container with the new environment.

## Using a local Ollama model from Docker

The app's default Ollama endpoint when running natively is
`http://127.0.0.1:11434`. This address **does not point to the host** when
used inside a container. Compose instead sets:

```dotenv
OLLAMA_HOST=http://host.docker.internal:11434
```

The app respects `OLLAMA_HOST` in both the CLI provider session and provider
model discovery. On Docker Desktop for Windows and macOS,
`host.docker.internal` resolves to the host computer. On Linux with a
supported Docker Engine, Compose adds a `host-gateway` host mapping.

Start the Ollama service on your computer and download a model, for example:

```powershell
ollama pull llama3.2:1b
ollama list
```

If Docker cannot connect to Ollama, check that the server is running and
reachable from Docker Desktop. Some host configurations require Ollama to
listen on an address accessible to containers rather than host loopback only
(e.g. configure the Ollama **server process** with
`OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama). Restrict network
exposure appropriately; the Ollama HTTP API has no built-in authentication.

If your Ollama service runs on another address, set the full HTTP URL in the
Compose `.env` file and recreate the UI service:

```dotenv
OLLAMA_HOST=http://host.docker.internal:11434
```

Use an accessible address in place of the example. `OLLAMA_HOST` in the Compose
file refers to the **destination URL**; it is not the server's listen setting.

## Persistence and custom configuration

- Reports and charts are shared to the host through `./results:/app/results`.
- `settings.toml` is bind-mounted read-only; edit it locally to change
  experiment defaults. The container can be restarted to load changes.
- Both CLI and UI use the same reports directory and Docker image.
- The app's bundled Streamlit theme is copied into `/app/.streamlit/`.

To change the host browser port, use in `.env`:

```dotenv
STREAMLIT_PORT=8502
```

Then recreate the UI container and visit **http://localhost:8502**.

## Debugging

If the page does not load:

```powershell
docker compose ps
docker compose logs --tail=100 ui
docker compose config
```

- **Port already in use:** Set `STREAMLIT_PORT=8502` in `.env`, then
  `docker compose up -d --force-recreate ui`.
- **No model found / Ollama connection refused:** Check
  `host.docker.internal:11434` reachability and that the requested model is installed.
- **Groq/OpenAI authentication error:** Verify your local `.env` file and
  recreate the container. Never paste API keys into issue reports or screenshots.
- **No previous experiment results:** Check `./results/` on the host; do not
  delete the directory when rebuilding.
- **Container marked unhealthy:** Inspect UI logs; the healthcheck calls
  Streamlit's internal `/_stcore/health` endpoint.

## Validation before merging

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs --tail=50 ui
docker compose run --rm sampling-lab --help
uv run pytest
```

Use `http://localhost:8501` to confirm that the sidebar, experiments,
comparison and saved results load. Testing against a real Ollama or cloud API
requires those services separately and is not part of the Docker image build.
