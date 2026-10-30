"""Proxy wiring for DSML recovery: response-side only, gated per model."""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager

MODEL = "deepseek-v4.1-flash"
LEAK = (
    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo_tool\n"
    '{"input": "hi"}<｜tool_call_end｜><｜tool_calls_end｜>'
)


def _client():
    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _stub_proxy(monkeypatch):
    monkeypatch.setattr(proxy.config, "MODELS", {MODEL: ["http://up"]})
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


def _respond(monkeypatch, *, content_type: str, chunks: list[bytes]):
    async def _chunks():
        for chunk in chunks:
            yield chunk

    async def _send(req, **kwargs):
        resp = httpx.Response(200, content=_chunks(), headers={"Content-Type": content_type})
        resp.request = req
        return resp

    monkeypatch.setattr(proxy.client, "send", _send)


def _post(stream: bool, model: str = MODEL):
    return _client().post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        headers={"Authorization": "Bearer good"},
    )


def _chat_body(content: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
    ).encode()


def test_non_streaming_leak_becomes_tool_calls(monkeypatch):
    _respond(monkeypatch, content_type="application/json", chunks=[_chat_body(LEAK)])
    resp = _post(stream=False)
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "echo_tool"


def test_non_streaming_clean_body_is_byte_identical(monkeypatch):
    body = _chat_body("Hello, world.")
    _respond(monkeypatch, content_type="application/json", chunks=[body])
    assert _post(stream=False).content == body


def test_streaming_leak_is_recovered_end_to_end(monkeypatch):
    def chunk(text=None, finish=None):
        obj = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": text} if text is not None else {}, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

    _respond(
        monkeypatch,
        content_type="text/event-stream",
        chunks=[chunk(LEAK[:40]), chunk(LEAK[40:]), chunk(None, "stop"), b"data: [DONE]\n\n"],
    )
    resp = _post(stream=True)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "tool_calls_begin" not in resp.text
    assert '"finish_reason":"tool_calls"' in resp.text
    assert '"name":"echo_tool"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_streaming_without_leak_is_byte_identical(monkeypatch):
    events = [
        (
            b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
            b'"choices":[{"index":0,"delta":{"content":"A long clean answer that crosses the sniff window."},'
            b'"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
    _respond(monkeypatch, content_type="text/event-stream", chunks=events)
    assert _post(stream=True).content == b"".join(events)


def test_other_models_are_untouched(monkeypatch):
    monkeypatch.setattr(proxy.config, "MODELS", {"qwen3.6-35b-a3b": ["http://up"]})
    body = _chat_body(LEAK)
    _respond(monkeypatch, content_type="application/json", chunks=[body])
    resp = _post(stream=False, model="qwen3.6-35b-a3b")
    assert resp.content == body
