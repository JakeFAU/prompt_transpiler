# Prompt Compiler (Transpiler)

> **Don't rewrite prompts. Compile them.**

A specialized transpiler that converts LLM prompts into model-specific formats (Intermediate Representation -> Optimized Output). It solves the "Prompt Drift" problem where a prompt optimized for GPT-4 fails on Gemini 1.5 or Claude 3.5.

## 🏗 Architecture

The system operates like a standard compiler toolchain:

1. **Decompile:** Analyzes a raw user prompt (and optional intent) to generate a model-agnostic **Intermediate Representation (IR)**.
2. **Transform:** Applies strict, model-specific heuristics (The "Instruction Set") defined in the `Registry`.
3. **Compile:** Generates the final, optimized prompt structure for the target runtime (e.g., XML for Claude, Markdown for Gemini).

## 🚀 Features

* **Model Agnostic IR:** Breaks prompts down into `Intent`, `Constraints`, `Context`, and `DataSchema`.
* **Strict Output Enforcement:** Handles the nuances of `response_format={"type": "json_schema"}` (OpenAI) vs `response_mime_type` (Gemini) vs Prefill-Injection (Claude).
* **Telemetry Native:** Built-in OpenTelemetry support for tracing prompt compilation pipelines.
* **Fail-Fast Config:** Validates API keys and provider availability at startup.

## 📦 Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repo
git clone [https://github.com/JakeFAU/prompt_transpiler.git](https://github.com/JakeFAU/prompt_transpiler.git)
cd prompt_transpiler

# Install dependencies
poetry install

# Activate the shell
poetry shell
````

## 🛠 Configuration

Create a `.env` file or set environment variables:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_API_KEY=hf_...

# Optional: Observability
USE_OPENTEL=true
OPENTEL_ENDPOINT=http://localhost:4317
```

## 🧑‍💻 Usage

### As a Library

```python
from prompt_compiler.registry import MODEL_REGISTRY
from prompt_compiler.compiler import Compiler

async def main():
    # 1. Initialize
    compiler = Compiler()

    # 2. Decompile (Understand the intent)
    raw_prompt = "Extract the stock tickers from this text and give me JSON."
    ir = await compiler.decompile(raw_prompt, source_model="human")

    # 3. Compile for Targets
    gpt_prompt = await compiler.compile(ir, target="gpt-4o")
    claude_prompt = await compiler.compile(ir, target="claude-3-5-sonnet")

    print(f"GPT Instruction:\n{gpt_prompt}")
    print(f"Claude Instruction:\n{claude_prompt}")
```

## 🛡 Development & Testing

We enforce strict typing and code quality.

```bash
# Run the full test suite
poetry run pytest

# Run type checks (Strict Mode)
poetry run mypy .

# Run pre-commit hooks manually
pre-commit run --all-files
```

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch.
3. Ensure `pre-commit` passes (Ruff + Mypy).
4. Submit a PR.
