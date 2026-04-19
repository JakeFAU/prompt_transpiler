.PHONY: install test lint format clean

install:
	uv sync --extra dev --extra test

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy src

format:
	uv run ruff format .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
