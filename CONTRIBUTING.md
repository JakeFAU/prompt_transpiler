# Contributing to prompt-compiler

Thank you for your interest in contributing to prompt-compiler!

## Development Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/prompt-compiler.git
    cd prompt-compiler
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Running Tests

We use `pytest` for testing. Ensure your changes pass all tests and maintain at least 85% coverage.

```bash
poetry run pytest
```

## Linting and Formatting

We use `ruff` for linting and formatting and `mypy` for static type checking.

```bash
# Lint
poetry run ruff check .

# Format
poetry run ruff format .

# Type check
poetry run mypy src
```

## Pull Requests

1.  Fork the repository and create a new branch for your feature or fix.
2.  Ensure all tests pass and coverage requirements are met.
3.  Ensure code is linted and formatted.
4.  Submit a Pull Request with a clear description of your changes.

## Documentation

Documentation is built using Sphinx. To build the docs locally:

```bash
poetry install --with docs
cd docs
poetry run sphinx-build -b html . _build/html
```
