import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api_keys import KeysManager
from src.config import config
from src.rate_limit import enforce_chat_key_rate_limit

pytestmark = pytest.mark.asyncio


class FakeRedis:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.expirations: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.expirations[key] = seconds

    async def ttl(self, key: str) -> int:
        return self.expirations.get(key, -1)


def _request(token: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def _seed_keys() -> None:
    manager = KeysManager()
    manager.keys = {"chat-token", "api-token"}
    manager.key_types = {"chat-token": "chat", "api-token": "api"}


async def test_chat_key_hits_minute_limit(monkeypatch):
    fake_redis = FakeRedis()
    _seed_keys()
    monkeypatch.setattr("src.rate_limit.get_redis", lambda: fake_redis)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_DAY", 100)

    await enforce_chat_key_rate_limit(_request("chat-token"))

    with pytest.raises(HTTPException) as exc:
        await enforce_chat_key_rate_limit(_request("chat-token"))

    assert exc.value.status_code == 429
    assert exc.value.headers == {"Retry-After": "60"}
    assert exc.value.detail["scope"] == "requests_per_minute"


async def test_non_chat_key_is_not_limited(monkeypatch):
    fake_redis = FakeRedis()
    _seed_keys()
    monkeypatch.setattr("src.rate_limit.get_redis", lambda: fake_redis)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_DAY", 0)

    await enforce_chat_key_rate_limit(_request("api-token"))

    assert fake_redis.counts == {}


async def test_non_positive_limits_are_disabled(monkeypatch):
    fake_redis = FakeRedis()
    _seed_keys()
    monkeypatch.setattr("src.rate_limit.get_redis", lambda: fake_redis)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_DAY", 0)

    await enforce_chat_key_rate_limit(_request("chat-token"))
    await enforce_chat_key_rate_limit(_request("chat-token"))

    assert fake_redis.counts == {}


async def test_search_limit_has_separate_daily_scope(monkeypatch):
    fake_redis = FakeRedis()
    _seed_keys()
    monkeypatch.setattr("src.rate_limit.get_redis", lambda: fake_redis)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_MINUTE", 100)
    monkeypatch.setattr(config, "CHAT_RATE_LIMIT_PER_DAY", 100)
    monkeypatch.setattr(config, "CHAT_SEARCH_RATE_LIMIT_PER_DAY", 1)

    await enforce_chat_key_rate_limit(_request("chat-token"), search=True)

    with pytest.raises(HTTPException) as exc:
        await enforce_chat_key_rate_limit(_request("chat-token"), search=True)

    assert exc.value.status_code == 429
    assert exc.value.headers == {"Retry-After": str(24 * 60 * 60)}
    assert exc.value.detail["scope"] == "search_requests_per_day"
