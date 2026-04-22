import redis.asyncio as redis

from src.config import config
from src.logger import setup_logger

logger = setup_logger(__name__)

KEY_PREFIX = "libertai:"


def k(*parts: str) -> str:
    return KEY_PREFIX + ":".join(parts)


_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            config.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
            health_check_interval=30,
        )
        logger.info(f"Redis client initialized ({config.REDIS_URL})")
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
