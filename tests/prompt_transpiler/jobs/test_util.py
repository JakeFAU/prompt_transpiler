import datetime

from prompt_transpiler.jobs.util import utc_now_iso


def test_utc_now_iso_returns_string():
    """Test that utc_now_iso returns a string."""
    result = utc_now_iso()
    assert isinstance(result, str)


def test_utc_now_iso_is_valid_iso8601():
    """Test that the returned string is a valid ISO-8601 format string."""
    result = utc_now_iso()
    # If this doesn't raise ValueError, it's a valid ISO-8601 string
    dt = datetime.datetime.fromisoformat(result)
    assert dt.tzinfo == datetime.UTC


def test_utc_now_iso_value(monkeypatch):
    """Test that utc_now_iso returns the correct formatted string for a specific datetime."""

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    monkeypatch.setattr("prompt_transpiler.jobs.util.datetime", MockDatetime)

    result = utc_now_iso()
    assert result == "2023-01-01T12:00:00+00:00"
