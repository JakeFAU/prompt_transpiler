# Prompt Compiler (Transpiler)

> **Don't rewrite prompts. Compile them.**

A specialized transpiler that converts LLM prompts into model-specific formats (Intermediate Representation -> Optimized Output). It solves the "Prompt Drift" problem where a prompt optimized for GPT-4 fails on Gemini or Claude.

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
* **Telemetry Native:** Built-in OpenTelemetry support for tracing prompt compilation pipelines.
* **Configurable Scoring:** Weighted scoring algorithm with customizable thresholds and retry logic.
* **Fail-Fast Config:** Validates API keys and provider availability at startup via Dynaconf.

## 📦 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and requires **Python 3.13+**.

```bash
# Clone the repo
git clone https://github.com/JakeFAU/prompt_transpiler.git
cd prompt_transpiler

# Install dependencies
poetry install

# Activate the shell
poetry shell
```

## 🛠 Configuration

Create a `.secrets.toml` file or set environment variables (prefixed with `PRCOMP_`):

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
export PRCOMP_OPENAI_API_KEY=sk-...
export PRCOMP_GEMINI_API_KEY=...
export PRCOMP_ANTHROPIC_API_KEY=sk-ant-...
export PRCOMP_HUGGINGFACE_API_KEY=hf_...
```

### Configuration Options

The `settings.toml` file controls pipeline behavior:

```toml
[default.compiler]
MAX_RETRIES = 3           # Maximum optimization attempts
SCORE_THRESHOLD = 0.8     # Minimum score to accept a prompt
EARLY_STOP_PATIENCE = 1   # Retries without improvement before stopping

[default.roles.architect]
PROVIDER = "openai"
MODEL = "gpt-4o"

[default.roles.decompiler]
PROVIDER = "gemini"
MODEL = "gemini-1.5-flash"

[default.roles.judge]
PROVIDER = "openai"
MODEL = "gpt-4o"
```

## 🧑‍💻 Usage

### CLI

```bash
# Basic usage: compile a prompt from one model to another
prompt-compiler "Extract the stock tickers from this text and give me JSON." \
    --source gpt-4o-mini \
    --target gemini-2.5-flash

# Compile from a file
prompt-compiler ./my_prompt.txt -s gpt-4o -t gemini-2.5-pro

# Save output to file
prompt-compiler "Your prompt here" -o compiled_prompt.txt

# Override agent models
prompt-compiler "Your prompt" \
    --architect-provider openai \
    --architect-model gpt-4o \
    --decompiler-provider gemini \
    --decompiler-model gemini-2.5-flash

# Adjust optimization parameters
prompt-compiler "Your prompt" --max-retries 5 --score-threshold 0.9

# Verbose output
prompt-compiler "Your prompt" -v

# Disable telemetry
prompt-compiler "Your prompt" --no-telemetry
```

### As a Library

```python
from prompt_compiler.core.pipeline import compile_pipeline

async def main():
    raw_prompt = "Extract the stock tickers from this text and give me JSON."

    result = await compile_pipeline(
        raw_prompt,
        source_model="gpt-4o-mini",
        target_model="gemini-2.5-flash",
        source_provider="openai",
        target_provider="gemini",
        max_retries=3,
        score_threshold=0.8,
    )

    print(f"Compiled Prompt:\n{result.prompt}")
```

### Advanced: Custom Pipeline

```python
from prompt_compiler.core.pipeline import PromptCompilerPipeline
from prompt_compiler.core.registry import ModelRegistry

async def main():
    # Create pipeline with custom settings
    pipeline = PromptCompilerPipeline(
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
poetry run pytest

# Run specific tests
poetry run pytest tests/prompt_compiler/core/test_pipeline.py

# Run type checks (Strict Mode)
poetry run mypy .

# Run linting
poetry run ruff check .

# Run pre-commit hooks manually
pre-commit run --all-files
```

## 📁 Project Structure

```bash
src/prompt_compiler/
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

## 📄 License

See [LICENSE](LICENSE) for details.
