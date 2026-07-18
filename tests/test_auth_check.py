import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api_keys import KeysManager
from src.auth import router


@pytest.fixture(autouse=True)
def _reset_keys_manager():
    """Snapshot and restore KeysManager singleton state around each test."""
    manager = KeysManager()
    saved_keys = manager.keys.copy()
    saved_invalid = manager.invalid_keys.copy()
    yield
    manager.keys = saved_keys
    manager.invalid_keys = saved_invalid


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_valid_key_ok():
    KeysManager().keys = {"good"}
    KeysManager().invalid_keys = {}
    resp = _client().get("/libertai/auth/check", headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200


def test_key_in_both_sets_treated_as_valid():
    # Defensive overlap case: valid set wins (lists are disjoint by construction).
    KeysManager().keys = {"both"}
    KeysManager().invalid_keys = {"both": {"reason": "no_credits", "message": "No credits."}}
    resp = _client().get("/libertai/auth/check", headers={"Authorization": "Bearer both"})
    assert resp.status_code == 200


def test_blocked_key_403_with_reason():
    KeysManager().keys = set()
    KeysManager().invalid_keys = {"blocked": {"reason": "no_credits", "message": "No credits."}}
    resp = _client().get("/libertai/auth/check", headers={"Authorization": "Bearer blocked"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "no_credits"


def test_unknown_key_401():
    KeysManager().keys = set()
    KeysManager().invalid_keys = {}
    resp = _client().get("/libertai/auth/check", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401
