import json
from unittest import mock

import pytest

from prompt_transpiler.jobs.util import json_loads, parse_int_env


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


@mock.patch("os.getenv")
def test_parse_int_env_valid(mock_getenv):
    mock_getenv.return_value = "42"
    default_val = 10
    expected_val = 42
    assert parse_int_env("TEST_INT", default_val) == expected_val


@mock.patch("os.getenv")
def test_parse_int_env_none(mock_getenv):
    mock_getenv.return_value = None
    default_val = 10
    assert parse_int_env("TEST_INT", default_val) == default_val


@mock.patch("os.getenv")
def test_parse_int_env_empty(mock_getenv):
    mock_getenv.return_value = "   "
    default_val = 10
    assert parse_int_env("TEST_INT", default_val) == default_val


@mock.patch("os.getenv")
def test_parse_int_env_invalid_value_error(mock_getenv):
    mock_getenv.return_value = "not_an_int"
    default_val = 10
    assert parse_int_env("TEST_INT", default_val) == default_val
