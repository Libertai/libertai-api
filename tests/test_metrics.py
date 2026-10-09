"""Tests for the Prometheus instrumentation (src/metrics.py + proxy point increments)."""

import httpx
import pytest
from fastapi.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager
from src.config import config
from src.server import app

TOKEN = "test-token"


def _client():
    return TestClient(app)


@pytest.fixture
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


def _stub_forwarding(monkeypatch, models=None) -> list[str]:
    """Stub the upstream so an admitted request fails over with a connect error.

    Returns the list of send urls attempted (empty when the gate rejected the
    request before any upstream attempt).
    """
    monkeypatch.setattr(proxy.config, "MODELS", models or {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _loads(_model):
        return {"http://up": 50}

    async def _noop(*args, **kwargs):
        return None

    sends: list[str] = []

    async def _refuse(req, **kwargs):
        sends.append(str(req.url))
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)
    return sends


def _gate_thresholds(monkeypatch):
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)


def _post_with_key(key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _client().post("/v1/chat/completions", json={"model": "m"}, headers=headers)


def _scrape(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", params={"token": TOKEN})
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    return resp.text


def test_metrics_disabled_when_token_unset(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", "")
    resp = _client().get("/metrics")
    assert resp.status_code == 401


def test_metrics_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", params={"token": "wrong"}).status_code == 401


def test_metrics_not_exposed_in_openapi_docs():
    resp = _client().get("/openapi.json")
    assert resp.status_code == 200
    assert "/metrics" not in resp.json()["paths"]


def test_metrics_scrape_lists_all_metric_families(monkeypatch):
    text = _scrape(monkeypatch)
    for family in (
        "libertai_http_requests_total",
        "libertai_http_request_duration_seconds",
        "libertai_requests_total",
        "libertai_gate_rejections_total",
        "libertai_gate_wait_seconds",
        "libertai_upstream_failures_total",
        "libertai_upstream_request_duration_seconds",
    ):
        assert f"# HELP {family}" in text


def test_transport_metrics_count_a_request(monkeypatch):
    # /health without the lifespan running answers 503 "starting".
    assert _client().get("/health").status_code == 503
    text = _scrape(monkeypatch)
    assert 'libertai_http_requests_total{endpoint="health",method="GET",status="503"}' in text


def test_gate_rejection_and_tier_metrics(monkeypatch, _reset_keys_manager):
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_overloaded"
    assert sends == []

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="free"}' in text
    assert 'libertai_gate_rejections_total{model="m"}' in text


def test_admitted_request_counts_upstream_failure(monkeypatch, _reset_keys_manager):
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    KeysManager().keys = {"paid"}
    KeysManager().tiers = {"paid": "pro"}

    resp = _post_with_key("paid")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "All servers unavailable for model m"
    assert len(sends) == 1

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="paid"}' in text
    assert 'libertai_upstream_failures_total{model="m",server="http://up"}' in text
