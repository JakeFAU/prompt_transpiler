import pytest
import structlog
from typing import Generator

from prompt_transpiler.utils.logging import clear_context, set_context

@pytest.fixture(autouse=True)
def clean_contextvars() -> Generator[None, None, None]:
    """Ensure contextvars are cleared before and after each test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()

def test_clear_context():
    # Set up some context variables
    set_context(key1="value1", key2="value2")

    # Check that they are set
    context = structlog.contextvars.get_contextvars()
    assert context == {"key1": "value1", "key2": "value2"}

    # Clear specific key
    clear_context("key1")

    # Check that key1 is removed and key2 remains
    context = structlog.contextvars.get_contextvars()
    assert context == {"key2": "value2"}

def test_clear_context_no_keys():
    # Set up some context variables
    set_context(key1="value1", key2="value2")

    # Check that they are set
    context = structlog.contextvars.get_contextvars()
    assert context == {"key1": "value1", "key2": "value2"}

    # Clear all keys (no arguments provided)
    clear_context()

    # Check that context is empty
    context = structlog.contextvars.get_contextvars()
    assert context == {}
