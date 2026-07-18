import json

import pytest

from src.api_keys import KeysManager, parse_snapshot


@pytest.fixture(autouse=True)
def _reset_keys_manager():
    """Snapshot and restore KeysManager singleton state around each test."""
    manager = KeysManager()
    saved_keys = manager.keys.copy()
    saved_invalid = manager.invalid_keys.copy()
    yield
    manager.keys = saved_keys
    manager.invalid_keys = saved_invalid


def test_parse_new_dict_shape():
    raw = json.dumps({"keys": ["a", "b"], "invalid_keys": {"c": {"reason": "expired", "message": "m"}}})
    keys, invalid = parse_snapshot(raw)
    assert keys == {"a", "b"}
    assert invalid == {"c": {"reason": "expired", "message": "m"}}


def test_parse_legacy_list_shape():
    keys, invalid = parse_snapshot(json.dumps(["a", "b"]))
    assert keys == {"a", "b"}
    assert invalid == {}


def test_parse_dict_shape_missing_invalid_field():
    keys, invalid = parse_snapshot(json.dumps({"keys": ["a"]}))
    assert keys == {"a"}
    assert invalid == {}


def test_key_invalid_info_lookup():
    manager = KeysManager()
    manager.keys = {"good"}
    manager.invalid_keys = {"blocked": {"reason": "no_credits", "message": "m"}}
    assert manager.key_invalid_info("blocked") == {"reason": "no_credits", "message": "m"}
    assert manager.key_invalid_info("good") is None
    assert manager.key_invalid_info("unknown") is None
