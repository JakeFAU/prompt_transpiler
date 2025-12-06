.PHONY: install test lint format clean

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .
	poetry run mypy src

format:
	poetry run ruff format .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
