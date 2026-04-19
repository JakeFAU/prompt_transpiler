# Contributing to prompt-transpiler

Thank you for your interest in contributing to prompt-transpiler!

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/JakeFAU/prompt_transpiler.git
    cd prompt_transpiler
    ```

2. **Install dependencies:**

    ```bash
    uv sync --extra dev --extra test
    ```

3. **Run commands in the project environment:**

    ```bash
    uv run prompt-transpiler --help
    ```

## Running Tests

We use `pytest` for testing. Ensure your changes pass all tests and maintain at least 85% coverage.

```bash
uv run pytest
```

## Linting and Formatting

We use `ruff` for linting and formatting and `mypy` for static type checking.

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src
```

## Pull Requests

1. Fork the repository and create a new branch for your feature or fix.
2. Ensure all tests pass and coverage requirements are met.
3. Ensure code is linted and formatted.
4. Submit a Pull Request with a clear description of your changes.

## Documentation

Documentation is built using Sphinx. To build the docs locally:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```
