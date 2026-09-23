import json

import pytest

from src.api_keys import KeysManager, parse_snapshot


@pytest.fixture(autouse=True)
def _reset_keys_manager():
    """Snapshot and restore KeysManager singleton state around each test."""
    manager = KeysManager()
    saved_keys = manager.keys.copy()
    saved_invalid = manager.invalid_keys.copy()
    saved_tiers = manager.tiers.copy()
    yield
    manager.keys = saved_keys
    manager.invalid_keys = saved_invalid
    manager.tiers = saved_tiers


def test_parse_new_dict_shape():
    raw = json.dumps(
        {
            "keys": ["a", "b"],
            "invalid_keys": {"c": {"reason": "expired", "message": "m"}},
            "tiers": {"a": "free", "b": "pro", "c": "free"},
        }
    )
    keys, invalid, tiers = parse_snapshot(raw)
    assert keys == {"a", "b"}
    assert invalid == {"c": {"reason": "expired", "message": "m"}}
    assert tiers == {"a": "free", "b": "pro", "c": "free"}


def test_parse_legacy_list_shape():
    keys, invalid, tiers = parse_snapshot(json.dumps(["a", "b"]))
    assert keys == {"a", "b"}
    assert invalid == {}
    assert tiers == {}


def test_parse_dict_shape_missing_invalid_field():
    keys, invalid, tiers = parse_snapshot(json.dumps({"keys": ["a"]}))
    assert keys == {"a"}
    assert invalid == {}
    assert tiers == {}


def test_key_invalid_info_lookup():
    manager = KeysManager()
    manager.keys = {"good"}
    manager.invalid_keys = {"blocked": {"reason": "no_credits", "message": "m"}}
    assert manager.key_invalid_info("blocked") == {"reason": "no_credits", "message": "m"}
    assert manager.key_invalid_info("good") is None
    assert manager.key_invalid_info("unknown") is None


def test_key_tier_lookup():
    manager = KeysManager()
    manager.tiers = {"free_key": "free", "pro_key": "pro"}
    assert manager.tier("free_key") == "free"
    assert manager.tier("pro_key") == "pro"
    assert manager.tier("unknown") is None
