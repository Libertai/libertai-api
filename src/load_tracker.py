from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

LOAD_KEY = k("inflight_load")
# Hash expires after this many seconds of no traffic — self-cleanup on total idle.
LOAD_TTL = 600


async def get_all_loads() -> dict[str, int]:
    try:
        data = await get_redis().hgetall(LOAD_KEY)
        return {url: int(v) for url, v in data.items()}
    except Exception as e:
        logger.error(f"Failed to read inflight loads from Redis: {e}", exc_info=True)
        return {}


async def adjust(server: str, delta: int) -> None:
    if delta == 0:
        return
    try:
        r = get_redis()
        async with r.pipeline(transaction=False) as pipe:
            pipe.hincrby(LOAD_KEY, server, delta)
            pipe.expire(LOAD_KEY, LOAD_TTL)
            await pipe.execute()
    except Exception as e:
        logger.error(f"Failed to adjust inflight load for {server} by {delta}: {e}", exc_info=True)
