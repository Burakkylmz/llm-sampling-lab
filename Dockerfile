
# =====================================
# Stage 1: Build the Python environment
# =====================================

FROM python:3.12-slim AS builder

# Pin the uv version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies separately for Docker layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project \
    --no-editable

# Copy application source files
COPY README.md ./
COPY src/ ./src/

# Install application as a non-editable package
RUN uv sync \
    --locked \
    --no-dev \
    --no-editable


# =====================================
# Stage 2: Runtime image
# =====================================

FROM python:3.12-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Copy only the installed Python environment
COPY --from=builder /app/.venv /app/.venv

# Copy runtime configuration
COPY settings.toml ./settings.toml

# Directory for experiment reports and visualizations
RUN mkdir -p /app/results

# Run the installed CLI
ENTRYPOINT ["sampling-lab"]
