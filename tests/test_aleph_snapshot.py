import asyncio
import json

import src.aleph as aleph_module
from src.aleph import AlephService


class _FakeRedis:
    def __init__(self, raw=None):
        self._raw = raw

    async def get(self, key):
        return self._raw

    async def set(self, *args, **kwargs):
        return True


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        return _FakeResponse(self._payload)


def _sync(monkeypatch, raw):
    service = AlephService()
    monkeypatch.setattr(aleph_module, "get_redis", lambda: _FakeRedis(raw))
    asyncio.run(service.sync_from_redis())
    return service


def test_sync_from_redis_loads_models(monkeypatch):
    service = _sync(monkeypatch, json.dumps({"models": {"glm-5.3": {"id": "glm-5.3"}}}))
    assert service.models_loaded
    assert "glm-5.3" in service.models


def test_sync_from_redis_marks_authoritatively_empty_models_loaded(monkeypatch):
    service = _sync(monkeypatch, json.dumps({"models": {}}))
    assert service.models_loaded
    assert service.models == {}


def test_sync_from_redis_without_models_key_keeps_state(monkeypatch):
    """A snapshot from a pre-'models' release must not mark metadata as loaded."""
    service = _sync(monkeypatch, json.dumps({"redirections": {}}))
    assert not service.models_loaded
    assert service.models == {}


def test_refresh_marks_models_loaded(monkeypatch):
    payload = {
        "data": {
            "LTAI_PRICING": {
                "models": [{"id": "GLM-5.3", "capabilities": {"text": {"reasoning": True}}}],
                "redirections": [{"from": "alias", "to": "GLM-5.3"}],
            }
        }
    }
    service = AlephService()
    monkeypatch.setattr(aleph_module.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(payload))
    monkeypatch.setattr(aleph_module, "get_redis", lambda: _FakeRedis())
    asyncio.run(service.refresh())
    assert service.models_loaded
    assert "glm-5.3" in service.models
    assert service.is_reasoning_model("glm-5.3")
    assert service.resolve("alias") == "glm-5.3"
