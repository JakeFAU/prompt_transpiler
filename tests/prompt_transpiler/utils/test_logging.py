import pytest
import structlog
from prompt_transpiler.utils.logging import set_context


@pytest.fixture(autouse=True)
def clean_context():
    """Ensure context is clean before and after each test."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


def test_set_context():
    """Test that set_context correctly binds context variables."""
    # Ensure initially empty
    assert structlog.contextvars.get_contextvars() == {}

    # Set some context
    set_context(request_id="abc123", user_id="u-42")

    # Verify context
    ctx = structlog.contextvars.get_contextvars()
    assert ctx == {"request_id": "abc123", "user_id": "u-42"}

    # Add more context
    set_context(action="test")
    ctx = structlog.contextvars.get_contextvars()
    assert ctx == {"request_id": "abc123", "user_id": "u-42", "action": "test"}

    # Overwrite context
    set_context(user_id="u-99")
    ctx = structlog.contextvars.get_contextvars()
    assert ctx == {"request_id": "abc123", "user_id": "u-99", "action": "test"}
