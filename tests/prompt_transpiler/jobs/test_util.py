"""Tests for shared utility functions."""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest

from prompt_transpiler.jobs.util import (
    generate_job_id,
    json_dumps,
    json_loads,
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    run_coroutine_sync,
    sleep_ms,
    utc_now_iso,
)


def test_utc_now_iso() -> None:
    """Test generating UTC now in ISO-8601 format."""
    now = utc_now_iso()
    assert isinstance(now, str)
    # Ensure it's parseable as an ISO string
    dt = datetime.fromisoformat(now)
    assert dt.tzinfo is not None


def test_generate_job_id() -> None:
    """Test generating a random job ID."""
    job_id = generate_job_id()
    assert isinstance(job_id, str)
    assert len(job_id) == 32
    # Ensure it's valid hex
    int(job_id, 16)


def test_json_dumps() -> None:
    """Test JSON serialization."""
    assert json_dumps({"a": 1, "b": "test"}) == '{"a":1,"b":"test"}'

    # Test fallback to str() for complex objects
    dt = datetime(2023, 1, 1, 12, 0, 0)
    assert json_dumps({"dt": dt}) == '{"dt":"2023-01-01 12:00:00"}'


def test_json_loads() -> None:
    """Test JSON deserialization."""
    assert json_loads('{"a":1,"b":"test"}') == {"a": 1, "b": "test"}
    assert json_loads(None) is None
    assert json_loads("") is None


def test_parse_bool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing boolean environment variables."""
    # Test missing env var (uses default)
    monkeypatch.delenv("TEST_BOOL", raising=False)
    assert parse_bool_env("TEST_BOOL", True) is True
    assert parse_bool_env("TEST_BOOL", False) is False

    # Test truthy values
    for val in ["1", "true", "yes", "y", "on", "TRUE", "Yes", "  1  "]:
        monkeypatch.setenv("TEST_BOOL", val)
        assert parse_bool_env("TEST_BOOL", False) is True

    # Test falsy values
    for val in ["0", "false", "no", "n", "off", "FALSE", "No", "  0  ", "random"]:
        monkeypatch.setenv("TEST_BOOL", val)
        assert parse_bool_env("TEST_BOOL", True) is False


def test_parse_int_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing integer environment variables."""
    # Test missing env var
    monkeypatch.delenv("TEST_INT", raising=False)
    assert parse_int_env("TEST_INT", 42) == 42

    # Test empty string
    monkeypatch.setenv("TEST_INT", "   ")
    assert parse_int_env("TEST_INT", 42) == 42

    # Test valid ints
    monkeypatch.setenv("TEST_INT", "100")
    assert parse_int_env("TEST_INT", 42) == 100
    monkeypatch.setenv("TEST_INT", "-5")
    assert parse_int_env("TEST_INT", 42) == -5

    # Test invalid int
    monkeypatch.setenv("TEST_INT", "not_an_int")
    assert parse_int_env("TEST_INT", 42) == 42


def test_parse_float_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing float environment variables."""
    # Test missing env var
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert parse_float_env("TEST_FLOAT", 3.14) == 3.14

    # Test empty string
    monkeypatch.setenv("TEST_FLOAT", "   ")
    assert parse_float_env("TEST_FLOAT", 3.14) == 3.14

    # Test valid floats
    monkeypatch.setenv("TEST_FLOAT", "100.5")
    assert parse_float_env("TEST_FLOAT", 3.14) == 100.5
    monkeypatch.setenv("TEST_FLOAT", "-5.25")
    assert parse_float_env("TEST_FLOAT", 3.14) == -5.25
    monkeypatch.setenv("TEST_FLOAT", "10")
    assert parse_float_env("TEST_FLOAT", 3.14) == 10.0

    # Test invalid float
    monkeypatch.setenv("TEST_FLOAT", "not_a_float")
    assert parse_float_env("TEST_FLOAT", 3.14) == 3.14


@patch("prompt_transpiler.jobs.util.time.sleep")
def test_sleep_ms(mock_sleep) -> None:
    """Test sleep in milliseconds."""
    sleep_ms(1500)
    mock_sleep.assert_called_once_with(1.5)


def test_run_coroutine_sync() -> None:
    """Test running an async coroutine from sync code."""

    async def simple_coro() -> int:
        await asyncio.sleep(0.01)
        return 42

    # Test with no existing event loop
    with patch("asyncio.get_event_loop", side_effect=RuntimeError):
        result = run_coroutine_sync(simple_coro())
        assert result == 42

    # Test with existing but not running event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = run_coroutine_sync(simple_coro())
        assert result == 42
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    # Test with running event loop (mocking the loop to hit the coverage branch)
    with patch("asyncio.get_event_loop") as mock_get_event_loop:
        mock_loop = mock_get_event_loop.return_value
        mock_loop.is_running.return_value = True

        with patch("asyncio.new_event_loop") as mock_new_event_loop:
            mock_new_loop = mock_new_event_loop.return_value
            mock_new_loop.run_until_complete.return_value = 42

            result = run_coroutine_sync(simple_coro())
            assert result == 42
