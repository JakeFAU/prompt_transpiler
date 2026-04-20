import json

import pytest

from prompt_transpiler.jobs.util import json_loads, parse_bool_env, parse_float_env, parse_int_env


def test_json_loads_none():
    assert json_loads(None) is None


def test_json_loads_empty_string():
    assert json_loads("") is None


def test_json_loads_valid_json_dict():
    assert json_loads('{"key": "value"}') == {"key": "value"}


def test_json_loads_valid_json_list():
    assert json_loads('["a", "b", "c"]') == ["a", "b", "c"]


def test_json_loads_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        json_loads('{"key": ')


def test_parse_int_env(monkeypatch):
    expected = 42
    assert parse_int_env("MISSING", expected) == expected
    monkeypatch.setenv("EMPTY", "")
    expected = 42
    assert parse_int_env("EMPTY", expected) == expected
    monkeypatch.setenv("VALID", "100")
    expected = 100
    assert parse_int_env("VALID", 42) == expected
    monkeypatch.setenv("INVALID", "abc")
    expected = 42
    assert parse_int_env("INVALID", expected) == expected


def test_parse_float_env(monkeypatch):
    expected = 3.14
    assert parse_float_env("MISSING", expected) == expected
    monkeypatch.setenv("EMPTY", "")
    expected = 3.14
    assert parse_float_env("EMPTY", expected) == expected
    monkeypatch.setenv("VALID", "2.71")
    expected = 2.71
    assert parse_float_env("VALID", 3.14) == expected
    monkeypatch.setenv("INVALID", "abc")
    expected = 3.14
    assert parse_float_env("INVALID", expected) == expected


def test_parse_bool_env(monkeypatch):
    assert parse_bool_env("MISSING", True) is True
    assert parse_bool_env("MISSING", False) is False

    monkeypatch.setenv("TRUE_1", "1")
    assert parse_bool_env("TRUE_1", False) is True
    monkeypatch.setenv("TRUE_TRUE", "true")
    assert parse_bool_env("TRUE_TRUE", False) is True
    monkeypatch.setenv("TRUE_YES", "yes")
    assert parse_bool_env("TRUE_YES", False) is True
    monkeypatch.setenv("TRUE_Y", "y")
    assert parse_bool_env("TRUE_Y", False) is True
    monkeypatch.setenv("TRUE_ON", "on")
    assert parse_bool_env("TRUE_ON", False) is True

    monkeypatch.setenv("FALSE_0", "0")
    assert parse_bool_env("FALSE_0", True) is False
    monkeypatch.setenv("FALSE_FALSE", "false")
    assert parse_bool_env("FALSE_FALSE", True) is False
    monkeypatch.setenv("FALSE_NO", "no")
    assert parse_bool_env("FALSE_NO", True) is False
