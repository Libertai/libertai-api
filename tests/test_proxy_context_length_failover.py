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


def test_context_length_400_fails_over_to_the_next_server(monkeypatch):
    # Force the small server first so the retry is the only way to reach the large one.
    monkeypatch.setattr(proxy.random, "sample", lambda urls, n: list(urls))
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
    monkeypatch.setattr(proxy.random, "sample", lambda urls, n: list(urls))
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
    monkeypatch.setattr(proxy.random, "sample", lambda urls, n: list(urls))
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
