"""Failover when a replica refuses a prompt for exceeding its context window."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager

_TOO_LONG = (
    b'{"error":{"message":"This model\'s maximum context length is 200000 tokens. However, you '
    b'requested 32768 output tokens and your prompt contains at least 167233 input tokens.",'
    b'"type":"BadRequestError","param":"input_tokens","code":400}}'
)


def _client():
    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _stub_proxy(monkeypatch):
    """Two registered servers, no Redis, a key that clears the gate."""
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://small", "http://large"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)
    monkeypatch.setattr(proxy.aleph_service, "is_reasoning_model", lambda model: False)
    monkeypatch.setattr(proxy.aleph_service, "is_vision_model", lambda model: True)

    async def _no_loads(_model):
        return {}

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(proxy, "get_model_loads", _no_loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)

    manager = KeysManager()
    saved_keys, saved_invalid = manager.keys.copy(), manager.invalid_keys.copy()
    manager.keys = {"good"}
    manager.invalid_keys = {}
    yield
    manager.keys, manager.invalid_keys = saved_keys, saved_invalid


def _streamed(status: int, body: bytes) -> httpx.Response:
    """A response whose body is still unread, as client.send(stream=True) returns."""

    async def _chunks():
        yield body

    return httpx.Response(status, content=_chunks(), headers={"Content-Type": "application/json"})


def _responder(monkeypatch, by_host: dict[str, httpx.Response]):
    """Serve a canned response per upstream host, recording the call order."""
    tried: list[str] = []

    async def _send(req, **kwargs):
        tried.append(req.url.host)
        resp = by_host[req.url.host]
        resp.request = req
        return resp

    monkeypatch.setattr(proxy.client, "send", _send)
    return tried


def _post():
    return _client().post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer good"},
    )


def test_tie_load_prefers_first_configured_server(monkeypatch):
    # Equal (here: unreported) loads keep models.json order: the first configured
    # server is tried first, and no failover happens when it succeeds.
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(200, b'{"ok":true}'),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small"]
    assert resp.status_code == 200


def test_healthy_tier_tie_prefers_first_configured_server(monkeypatch):
    # Production path: both servers healthy (so tiered first), equal loads ->
    # models.json order puts the small server in front.
    monkeypatch.setattr(
        proxy.server_health_monitor, "healthy_model_urls", {"m": ["http://small", "http://large"]}
    )
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(200, b'{"ok":true}'),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small"]
    assert resp.status_code == 200


def test_healthy_tier_load_still_reorders(monkeypatch):
    # Stable ordering must not defeat load balancing: the busy primary yields to
    # the idle overflow server within the healthy tier.
    monkeypatch.setattr(
        proxy.server_health_monitor, "healthy_model_urls", {"m": ["http://small", "http://large"]}
    )

    async def _loads(_model):
        return {"http://small": 1, "http://large": 0}

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(200, b'{"ok":true}'),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["large"]
    assert resp.status_code == 200


def test_busy_primary_stays_primary_below_capacity(monkeypatch):
    # A server's capacity is how many concurrent requests it absorbs before the
    # gateway overflows: with capacity 32, one (or 31) in-flight requests still
    # count as saturation 0, so the primary keeps taking traffic.
    monkeypatch.setattr(
        proxy.server_health_monitor, "healthy_model_urls", {"m": ["http://small", "http://large"]}
    )
    monkeypatch.setattr(proxy.config, "SERVER_CAPACITY", {"http://small": 32, "http://large": 12})

    async def _loads(_model):
        return {"http://small": 31, "http://large": 0}

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(200, b'{"ok":true}'),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small"]
    assert resp.status_code == 200


def test_saturated_primary_overflows(monkeypatch):
    # At capacity (32 in flight on a capacity-32 primary), saturation reaches 1
    # and the next request goes to the idle secondary.
    monkeypatch.setattr(
        proxy.server_health_monitor, "healthy_model_urls", {"m": ["http://small", "http://large"]}
    )
    monkeypatch.setattr(proxy.config, "SERVER_CAPACITY", {"http://small": 32, "http://large": 12})

    async def _loads(_model):
        return {"http://small": 32, "http://large": 0}

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(200, b'{"ok":true}'),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["large"]
    assert resp.status_code == 200


def test_context_length_400_fails_over_to_the_next_server(monkeypatch):
    # The small server is first in models.json, so the retry is the only way to
    # reach the large one.
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(400, _TOO_LONG),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small", "large"]
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_last_server_context_length_400_reaches_the_client(monkeypatch):
    # No replica can take the prompt: the client must get the real 400 and its
    # explanation, not the generic all-servers-unavailable 503.
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(400, _TOO_LONG),
            "large": _streamed(400, _TOO_LONG),
        },
    )

    resp = _post()

    assert tried == ["small", "large"]
    assert resp.status_code == 400
    assert "maximum context length" in resp.json()["error"]["message"]


def test_other_400s_are_returned_without_retrying(monkeypatch):
    # An unsupported parameter fails identically everywhere; retrying only doubles
    # the load and delays the error.
    body = b'{"error":{"message":"unknown field: temperatur","type":"BadRequestError","code":400}}'
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(400, body),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small"]
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "unknown field: temperatur"


_BOX_401 = b'{"detail":"Invalid API key"}'


def test_box_401_for_a_valid_key_fails_over(monkeypatch):
    # A box that has not received the key list yet must not fail a key we know is valid.
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(401, _BOX_401),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _post()

    assert tried == ["small", "large"]
    assert resp.status_code == 200


def test_box_401_for_an_unknown_key_is_returned(monkeypatch):
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(401, _BOX_401),
            "large": _streamed(200, b'{"ok":true}'),
        },
    )

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer unknown"},
    )

    assert tried == ["small"]
    assert resp.status_code == 401


def test_last_server_401_reaches_the_client(monkeypatch):
    tried = _responder(
        monkeypatch,
        {
            "small": _streamed(401, _BOX_401),
            "large": _streamed(401, _BOX_401),
        },
    )

    resp = _post()

    assert tried == ["small", "large"]
    assert resp.status_code == 401
