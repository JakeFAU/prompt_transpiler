import pytest
import structlog

from prompt_transpiler.utils.logging import clear_context, set_context


@pytest.fixture(autouse=True)
def _clean_contextvars():
    """Ensure contextvars are clear before and after each test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


def test_set_context():
    """Test setting context variables."""
    set_context(request_id="abc123", user_id="u-42")
    ctx = structlog.contextvars.get_contextvars()
    assert ctx.get("request_id") == "abc123"
    assert ctx.get("user_id") == "u-42"


def test_clear_context_specific_keys():
    """Test clearing specific context variables."""
    set_context(a="1", b="2", c="3")
    clear_context("a", "b")
    ctx = structlog.contextvars.get_contextvars()
    assert "a" not in ctx
    assert "b" not in ctx
    assert ctx.get("c") == "3"


def test_clear_context_all_keys():
    """Test clearing all context variables."""
    set_context(a="1", b="2")
    clear_context()
    ctx = structlog.contextvars.get_contextvars()
    assert "a" not in ctx
    assert "b" not in ctx
    assert not ctx
