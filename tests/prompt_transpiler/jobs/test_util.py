from prompt_transpiler.jobs.util import parse_float_env

def test_parse_float_env_valid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_ENV", "3.14")
    assert parse_float_env("TEST_FLOAT_ENV", 1.0) == 3.14

def test_parse_float_env_invalid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_ENV", "not_a_float")
    assert parse_float_env("TEST_FLOAT_ENV", 1.0) == 1.0

def test_parse_float_env_missing(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT_ENV", raising=False)
    assert parse_float_env("TEST_FLOAT_ENV", 1.0) == 1.0

def test_parse_float_env_empty(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_ENV", "")
    assert parse_float_env("TEST_FLOAT_ENV", 1.0) == 1.0

def test_parse_float_env_whitespace(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT_ENV", "   ")
    assert parse_float_env("TEST_FLOAT_ENV", 1.0) == 1.0
