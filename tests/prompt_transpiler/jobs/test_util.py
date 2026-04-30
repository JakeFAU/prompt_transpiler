import json

import pytest

from prompt_transpiler.jobs.store import _to_update_clause
from prompt_transpiler.jobs.util import json_loads


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


def test_to_update_clause_sql_injection():
    """Verify _to_update_clause raises ValueError for invalid column names."""
    match_str = "Invalid column name: invalid column; DROP TABLE compile_jobs; --"
    with pytest.raises(ValueError, match=match_str):
        _to_update_clause({"invalid column; DROP TABLE compile_jobs; --": 123})
