from prompt_transpiler.jobs.util import parse_float_env


def test_parse_float_env_success(monkeypatch):
    """Test parsing a valid float from environment."""
    expected_value = 3.14
    monkeypatch.setenv("TEST_FLOAT_ENV", "3.14")
    result = parse_float_env("TEST_FLOAT_ENV", 1.0)
    assert result == expected_value


def test_parse_float_env_missing(monkeypatch):
    """Test fallback when environment variable is missing."""
    monkeypatch.delenv("TEST_FLOAT_ENV", raising=False)
    result = parse_float_env("TEST_FLOAT_ENV", 1.0)
    assert result == 1.0

    monkeypatch.setenv("TEST_FLOAT_ENV", "")
    result = parse_float_env("TEST_FLOAT_ENV", 1.0)
    assert result == 1.0


def test_parse_float_env_invalid(monkeypatch):
    """Test fallback when environment variable is an invalid float."""
    monkeypatch.setenv("TEST_FLOAT_ENV", "not a float")
    result = parse_float_env("TEST_FLOAT_ENV", 1.0)
    assert result == 1.0
