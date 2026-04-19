import pytest
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
    import json
    with pytest.raises(json.JSONDecodeError):
        json_loads('{"key": ')
