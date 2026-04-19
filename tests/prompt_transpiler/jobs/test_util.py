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
    job_id_length = 32
    assert isinstance(job_id, str)
    assert len(job_id) == job_id_length
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
    default_int = 42
    valid_int_1 = 100
    valid_int_2 = -5

    # Test missing env var
    monkeypatch.delenv("TEST_INT", raising=False)
    assert parse_int_env("TEST_INT", default_int) == default_int

    # Test empty string
    monkeypatch.setenv("TEST_INT", "   ")
    assert parse_int_env("TEST_INT", default_int) == default_int

    # Test valid ints
    monkeypatch.setenv("TEST_INT", str(valid_int_1))
    assert parse_int_env("TEST_INT", default_int) == valid_int_1
    monkeypatch.setenv("TEST_INT", str(valid_int_2))
    assert parse_int_env("TEST_INT", default_int) == valid_int_2

    # Test invalid int
    monkeypatch.setenv("TEST_INT", "not_an_int")
    assert parse_int_env("TEST_INT", default_int) == default_int


def test_parse_float_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing float environment variables."""
    default_float = 3.14
    valid_float_1 = 100.5
    valid_float_2 = -5.25
    valid_float_3 = 10.0

    # Test missing env var
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert parse_float_env("TEST_FLOAT", default_float) == default_float

    # Test empty string
    monkeypatch.setenv("TEST_FLOAT", "   ")
    assert parse_float_env("TEST_FLOAT", default_float) == default_float

    # Test valid floats
    monkeypatch.setenv("TEST_FLOAT", str(valid_float_1))
    assert parse_float_env("TEST_FLOAT", default_float) == valid_float_1
    monkeypatch.setenv("TEST_FLOAT", str(valid_float_2))
    assert parse_float_env("TEST_FLOAT", default_float) == valid_float_2
    monkeypatch.setenv("TEST_FLOAT", "10")
    assert parse_float_env("TEST_FLOAT", default_float) == valid_float_3

    # Test invalid float
    monkeypatch.setenv("TEST_FLOAT", "not_a_float")
    assert parse_float_env("TEST_FLOAT", default_float) == default_float


@patch("prompt_transpiler.jobs.util.time.sleep")
def test_sleep_ms(mock_sleep) -> None:
    """Test sleep in milliseconds."""
    sleep_ms(1500)
    mock_sleep.assert_called_once_with(1.5)


def test_run_coroutine_sync() -> None:
    """Test running an async coroutine from sync code."""
    expected_result = 42

    async def simple_coro() -> int:
        await asyncio.sleep(0.01)
        return expected_result

    # Test with no existing event loop
    with patch("asyncio.get_event_loop", side_effect=RuntimeError):
        result = run_coroutine_sync(simple_coro())
        assert result == expected_result

    # Test with existing but not running event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = run_coroutine_sync(simple_coro())
        assert result == expected_result
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    # Test with running event loop (mocking the loop to hit the coverage branch)
    with patch("asyncio.get_event_loop") as mock_get_event_loop:
        mock_loop = mock_get_event_loop.return_value
        mock_loop.is_running.return_value = True

        with patch("asyncio.new_event_loop") as mock_new_event_loop:
            mock_new_loop = mock_new_event_loop.return_value
            mock_new_loop.run_until_complete.return_value = expected_result

            result = run_coroutine_sync(simple_coro())
            assert result == expected_result
