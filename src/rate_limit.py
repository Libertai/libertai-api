import hashlib
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from src.api_keys import KeysManager
from src.config import config
from src.redis_client import get_redis, k

CHAT_KEY_TYPE = "chat"
DAY_SECONDS = 24 * 60 * 60
MINUTE_SECONDS = 60


@dataclass(frozen=True)
class RateLimitRule:
    scope: str
    limit: int
    window_seconds: int


def extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _increment_or_raise(token: str, rule: RateLimitRule) -> None:
    if rule.limit <= 0:
        return

    redis = get_redis()
    redis_key = k("rate_limit", "chat", rule.scope, _token_hash(token))
    count = await redis.incr(redis_key)
    if count == 1:
        await redis.expire(redis_key, rule.window_seconds)
    if count <= rule.limit:
        return

    retry_after = await redis.ttl(redis_key)
    if retry_after is None or retry_after < 0:
        retry_after = rule.window_seconds
        await redis.expire(redis_key, retry_after)

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error": "rate_limit_exceeded",
            "scope": rule.scope,
            "limit": rule.limit,
            "window_seconds": rule.window_seconds,
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


async def enforce_chat_key_rate_limit(request: Request, *, search: bool = False) -> None:
    token = extract_bearer_token(request)
    if token is None:
        return

    keys_manager = KeysManager()
    if keys_manager.key_type_for(token) != CHAT_KEY_TYPE:
        return

    rules = [
        RateLimitRule("requests_per_minute", config.CHAT_RATE_LIMIT_PER_MINUTE, MINUTE_SECONDS),
        RateLimitRule("requests_per_day", config.CHAT_RATE_LIMIT_PER_DAY, DAY_SECONDS),
    ]
    if search:
        rules.insert(0, RateLimitRule("search_requests_per_day", config.CHAT_SEARCH_RATE_LIMIT_PER_DAY, DAY_SECONDS))

    for rule in rules:
        await _increment_or_raise(token, rule)
