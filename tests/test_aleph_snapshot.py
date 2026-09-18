import asyncio
import json

import src.aleph as aleph_module
from src.aleph import AlephService


class _FakeRedis:
    def __init__(self, raw):
        self._raw = raw

    async def get(self, key):
        return self._raw


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
