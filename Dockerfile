# syntax=docker/dockerfile:1.6
#
# Prompt Compiler API (APIFlask) + OpenTelemetry launcher
#
# - Uses Poetry for deps
# - Runs the API via "prompt-compiler-api"
# - Wraps runtime with: opentelemetry-instrument ...
# - Secrets are provided as environment variables (no files baked into image)
#
# Build:
#   docker build -t prompt-compiler:latest .
#
# Run (example):
#   docker run --rm -p 8080:8080 \
#     -e OPENAI_API_KEY=... \
#     -e GEMINI_API_KEY=... \
#     -e ANTHROPIC_API_KEY=... \
#     -e HUGGINGFACE_API_KEY=... \
#     -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
#     prompt-compiler:latest
#
# Docs:
#   http://localhost:8080/docs
#   http://localhost:8080/redoc
#   http://localhost:8080/openapi.json

############################
# 1) Builder
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps for building wheels (duckdb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry (pinned-ish)
ENV POETRY_VERSION=2.1.4 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install "poetry==${POETRY_VERSION}"

# Copy dependency manifests and README first for better layer caching
COPY pyproject.toml poetry.lock* README.md /app/

# Install runtime deps only (exclude dev/test/docs groups) without installing the project.
# If your Poetry version doesn't support --only main, use: --without dev,test,docs
RUN poetry install --only main --no-ansi --no-root

# Copy the rest of the source
COPY . /app

# Install the project package now that the source is present.
RUN poetry install --only main --no-ansi

############################
# 2) Runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed site-packages + scripts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app
RUN chmod +x /app/entrypoint.sh

# ---- Runtime configuration (container-friendly defaults) ----
# API server
ENV HOST=0.0.0.0 \
    PORT=8080 \
    PROMPT_COMPILER_ENV=prod \
    LOG_LEVEL=INFO

# Job store defaults (adjust to your implementation)
# If you end up using DuckDB for job persistence, ensure JOB_DB_PATH points to a writable path.
ENV JOB_STORE=duckdb \
    JOB_DB_PATH=/tmp/prompt_compiler_jobs.duckdb \
    WORKER_ENABLED=true

# ---- OpenTelemetry defaults ----
# You asked specifically to use opentelemetry-instrument launcher.
# NOTE: In OTEL, the endpoint usually includes scheme, e.g. http://collector:4317
# For gRPC, the default OTLP protocol is gRPC when using port 4317.
ENV OTEL_SERVICE_NAME=prompt-compiler-api \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_METRICS_EXPORTER=console \
    OTEL_LOGS_EXPORTER=none \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://0.0.0.0:4317 \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# ---- Secrets expected by the app (provided at runtime) ----
# Do NOT set values here. These are only documented to make expectations explicit.
#   OPENAI_API_KEY
#   GEMINI_API_KEY
#   ANTHROPIC_API_KEY
#   HUGGINGFACE_API_KEY

# Expose HTTP port (informational; still need -p)
EXPOSE 8080

# Healthcheck (adjust path if you change it)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % __import__('os').environ.get('PORT','8080')).read()" || exit 1

# Entrypoint selects OpenTelemetry based on PRCOMP_USE_OPENTELEMETRY/USE_OPENTELEMETRY.
ENTRYPOINT ["/app/entrypoint.sh"]
