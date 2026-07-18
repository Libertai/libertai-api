import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.proxy as proxy
from src.api_keys import KeysManager
from src.proxy import bearer_token


def test_bearer_token_strips_scheme_case_insensitively():
    assert bearer_token("Bearer abc123") == "abc123"
    assert bearer_token("bearer abc123") == "abc123"


def test_bearer_token_passthrough_without_scheme():
    assert bearer_token("abc123") == "abc123"


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
    app.include_router(proxy.router)
    return TestClient(app)


def test_blocked_key_gets_403_before_any_upstream_call(monkeypatch):
    # Register the model so the request clears the 404 model-resolution check
    # that runs before the key gate, without ever reaching an upstream server.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    KeysManager().invalid_keys = {"blocked": {"reason": "no_credits", "message": "No credits."}}

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m"},
        headers={"Authorization": "Bearer blocked"},
    )

    assert resp.status_code == 403
    assert resp.json() == {
        "error": {
            "message": "No credits.",
            "type": "invalid_request_error",
            "code": "no_credits",
        }
    }
