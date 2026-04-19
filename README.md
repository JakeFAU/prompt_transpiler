# Prompt Transpiler

> **Don't rewrite prompts. Transpile them.**

A specialized transpiler that converts LLM prompts into model-specific formats (Intermediate Representation -> Optimized Output). It solves the "Prompt Drift" problem where a prompt optimized for GPT-4 fails on Gemini or Claude.

This project is currently **alpha**. The APIs and scoring internals are still evolving, but the CLI, API, and core pipeline are stable enough for early adopters and contributors.

## 🏗 Architecture

The system operates like a standard compiler toolchain with specialized agents:

1. **Decompile:** The `Decompiler` agent analyzes a raw user prompt to generate a model-agnostic **Intermediate Representation (IR)**.
2. **Architect:** The `Architect` agent designs the optimal prompt structure for the target model.
3. **Judge:** The `Judge` agent scores candidate prompts using a weighted scoring algorithm.
4. **Pilot & Historian:** Supporting agents that manage execution flow and track optimization history.

### Supported Models

**OpenAI:** `gpt-5.1`, `gpt-5`, `gpt-5-pro`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`

**Gemini:** `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`

**Anthropic:** Claude models (dynamic registration)

**HuggingFace:** Local/hosted models via HuggingFace Hub

## 🚀 Features

* **Model Agnostic IR:** Breaks prompts down into `Intent`, `Constraints`, `Context`, and `DataSchema`.
* **Multi-Agent Pipeline:** Specialized agents (Decompiler, Architect, Judge, Pilot, Historian) collaborate for optimal results.
* **Strict Output Enforcement:** Handles the nuances of `response_format={"type": "json_schema"}` (OpenAI) vs `response_mime_type` (Gemini) vs Prefill-Injection (Claude).
* **Telemetry Native:** Built-in OpenTelemetry support for tracing prompt transpilation pipelines.
* **Configurable Scoring:** Pairwise judge-by-comparison is the default, with optional alternative scoring algorithms and retry logic.
* **Inspectable Runs:** Optional CLI score summaries and JSON reports include final scores, per-attempt history, token usage, and effective runtime settings.
* **Fail-Fast Config:** Validates API keys and provider availability at startup via Dynaconf.

## 📦 Installation

This project uses [uv](https://docs.astral.sh/uv/) and requires **Python 3.13+**.

### Install from PyPI

```bash
uv pip install prompt-transpiler

# CLI
uv run prompt-transpiler --help
```

### Install from source

```bash
git clone git@github.com:JakeFAU/prompt_transpiler.git
cd prompt_transpiler

uv sync --extra dev --extra test

uv run prompt-transpiler --help
```

## 🛠 Configuration

Create a `.secrets.toml` file or set environment variables (prefixed with `PRTRANS_`):

```bash
# .secrets.toml
[default]
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "..."
ANTHROPIC_API_KEY = "sk-ant-..."
HUGGINGFACE_API_KEY = "hf_..."
```

Or via environment variables:

```bash
export PRTRANS_OPENAI_API_KEY=sk-...
export PRTRANS_GEMINI_API_KEY=...
export PRTRANS_ANTHROPIC_API_KEY=sk-ant-...
export PRTRANS_HUGGINGFACE_API_KEY=hf_...
```

### Configuration Options

The `settings.toml` file controls pipeline behavior:

```toml
[default.transpiler]
MAX_RETRIES = 3           # Maximum optimization attempts
SCORE_THRESHOLD = 0.8     # Minimum score to accept a prompt
EARLY_STOP_PATIENCE = 1   # Retries without improvement before stopping
SCORING_ALGORITHM = "pairwise"

[default.roles.architect]
PROVIDER = "openai"
MODEL = "gpt-4o"

[default.roles.decompiler]
PROVIDER = "gemini"
MODEL = "gemini-2.5-flash"

[default.roles.judge]
PROVIDER = "openai"
MODEL = "gpt-4o"
```

## 🧑‍💻 Usage

### CLI

```bash
# Basic usage: transpile a prompt from one model to another
prompt-transpiler "Extract the stock tickers from this text and give me JSON." \
    --source gpt-4o-mini \
    --target gemini-2.5-flash

# The CLI will also print a semantic explanation of how the transpiled prompt differs
# from the original (diff agent). Disable with --quiet if you only want the prompt.

# Transpile from a file
prompt-transpiler ./my_prompt.txt -s gpt-4o -t gemini-2.5-pro

# Save output to file
prompt-transpiler "Your prompt here" -o transpiled_prompt.txt

# Override agent models
prompt-transpiler "Your prompt" \
    --architect-provider openai \
    --architect-model gpt-4o \
    --decompiler-provider gemini \
    --decompiler-model gemini-2.5-flash

# Adjust optimization parameters
prompt-transpiler "Your prompt" --max-retries 5 --score-threshold 0.9

# Override the default pairwise scoring strategy if needed
prompt-transpiler "Your prompt" --scoring-algo weighted

# Print a score summary and save a detailed machine-readable report
prompt-transpiler "Your prompt" \
    --show-scores \
    --report-json transpile-report.json

# Verbose output
prompt-transpiler "Your prompt" -v

# Disable telemetry
prompt-transpiler "Your prompt" --no-telemetry
```

### REST API

Run the API server from the project root:

```bash
# Local dev
prompt-transpiler-api

# Or with gunicorn
gunicorn -b 0.0.0.0:${PORT:-8080} prompt_transpiler.api.app:app
```

Docs:

* `http://localhost:8080/docs`
* `http://localhost:8080/redoc`
* `http://localhost:8080/openapi.json`

Environment variables:

* `HOST` (default `0.0.0.0`)
* `PORT` (default `8080`)
* `JOB_STORE` (`duckdb|sqlite|memory`, default `duckdb`)
* `JOB_DB_PATH` (default `/tmp/prompt_transpiler_jobs.duckdb`)
* `JOB_RETENTION_HOURS` (default `24`)
* `WORKER_ENABLED` (default `true`)
* `WORKER_POLL_INTERVAL_MS` (default `500`)
* `WORKER_CONCURRENCY` (default `1`)
* Provider secrets (OpenAI, Gemini, Anthropic, etc.) via env vars

Example API flow:

```bash
# Enqueue a transpile job
curl -X POST http://localhost:8080/v1/transpile-jobs \\
  -H "Content-Type: application/json" \\
  -d '{
    "raw_prompt": "Summarize this text into a bullet list.",
    "source_model": "gpt-4o-mini",
    "target_model": "gemini-2.5-flash"
  }'

# Check status
curl http://localhost:8080/v1/transpile-jobs/<job_id>

# Fetch result
curl http://localhost:8080/v1/transpile-jobs/<job_id>/result
```

### As a Library

```python
from prompt_transpiler.core.pipeline import transpile_pipeline

async def main():
    raw_prompt = "Extract the stock tickers from this text and give me JSON."

    result = await transpile_pipeline(
        raw_prompt,
        source_model_name="gpt-4o-mini",
        target_model_name="gemini-2.5-flash",
        source_provider="openai",
        target_provider="gemini",
        max_retries=3,
        score_threshold=0.8,
    )

    print(f"Transpiled Prompt:\n{result.prompt}")
    print(f"Attempts Recorded: {len(result.attempt_history)}")
```

### Advanced: Custom Pipeline

```python
from prompt_transpiler.core.pipeline import PromptTranspilerPipeline
from prompt_transpiler.core.registry import ModelRegistry

async def main():
    # Create pipeline with custom settings
    pipeline = PromptTranspilerPipeline(
        score_threshold=0.85,
        max_retries=5,
        early_stop_patience=2,
    )

    result = await pipeline.run(
        raw_prompt="Your complex prompt here",
        source_model="gpt-4o",
        target_model="gemini-2.5-pro",
        source_provider="openai",
        target_provider="gemini",
    )
```

## 🛡 Development & Testing

We enforce strict typing and code quality.

```bash
# Run the full test suite with coverage
uv run pytest

# Run specific tests
uv run pytest tests/prompt_transpiler/core/test_pipeline.py

# Run type checks (Strict Mode)
uv run mypy .

# Run linting
uv run ruff check .

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

## 📁 Project Structure

```bash
src/prompt_transpiler/
├── cli.py              # Command-line interface
├── config.py           # Dynaconf configuration loader
├── core/
│   ├── pipeline.py     # Main orchestration engine
│   ├── registry.py     # Model registry
│   ├── scoring.py      # Weighted scoring algorithm
│   ├── interfaces.py   # Abstract interfaces for agents
│   └── roles/          # Agent implementations
│       ├── architect.py
│       ├── decompiler.py
│       ├── historian.py
│       └── pilot.py
├── dto/
│   └── models.py       # Data transfer objects
├── llm/
│   ├── anthropic.py    # Anthropic adapter
│   ├── gemini.py       # Gemini adapter
│   ├── openai.py       # OpenAI adapter
│   ├── huggingface.py  # HuggingFace adapter
│   └── prompts/        # Prompt templates
└── utils/
    ├── logging.py      # Structured logging (structlog)
    └── telemetry.py    # OpenTelemetry instrumentation
```

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch.
3. Ensure `pre-commit` passes (Ruff + Mypy).
4. Submit a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for contributor and disclosure expectations.
