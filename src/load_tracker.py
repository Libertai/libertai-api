import time
from typing import Awaitable, cast

from src.config import config
from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

# A lease self-expires LEASE_TTL after its last (re)acquire, so a dropped release
# (cancelled request, killed process, Redis blip) can't leak load forever. The
# streaming path re-acquires every LEASE_REFRESH_INTERVAL to outlive long generations.
LEASE_TTL = 720
LEASE_REFRESH_INTERVAL = 300


def _key(server: str) -> str:
    return k("inflight", server)


def _all_servers() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for urls in config.MODELS.values():
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _prune_and_count(entries: dict[str, str], now: float) -> tuple[int, list[str]]:
    """Count live leases; return (load, expired_request_ids). Value is a "<deadline>"."""
    live = 0
    expired: list[str] = []
    for rid, val in entries.items():
        try:
            deadline = float(val)
        except (ValueError, TypeError):
            expired.append(rid)
            continue
        if deadline <= now:
            expired.append(rid)
        else:
            live += 1
    return live, expired


async def get_all_loads() -> dict[str, int]:
    servers = _all_servers()
    if not servers:
        return {}
    now = time.time()
    r = get_redis()
    try:
        async with r.pipeline(transaction=False) as pipe:
            for s in servers:
                pipe.hgetall(_key(s))
            raw = await pipe.execute()

        loads: dict[str, int] = {}
        to_prune: list[tuple[str, list[str]]] = []
        for s, entries in zip(servers, raw):
            live, expired = _prune_and_count(entries or {}, now)
            loads[s] = live
            if expired:
                to_prune.append((s, expired))

        if to_prune:
            async with r.pipeline(transaction=False) as pipe:
                for s, rids in to_prune:
                    pipe.hdel(_key(s), *rids)
                await pipe.execute()
        return loads
    except Exception as e:
        logger.error(f"Failed to read inflight loads from Redis: {e}", exc_info=True)
        return {}


async def acquire(server: str, request_id: str) -> None:
    try:
        r = get_redis()
        deadline = time.time() + LEASE_TTL
        async with r.pipeline(transaction=False) as pipe:
            pipe.hset(_key(server), request_id, f"{deadline}")
            pipe.expire(_key(server), LEASE_TTL + 60)
            await pipe.execute()
    except Exception as e:
        logger.error(f"Failed to acquire inflight lease for {server} ({request_id}): {e}", exc_info=True)


async def release(server: str, request_id: str) -> None:
    try:
        await cast("Awaitable[int]", get_redis().hdel(_key(server), request_id))
    except Exception as e:
        logger.error(f"Failed to release inflight lease for {server} ({request_id}): {e}", exc_info=True)
