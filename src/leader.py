import asyncio
import socket
import uuid
from typing import Awaitable, Callable

from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

LOCK_KEY = k("leader")
LOCK_TTL = 30
RENEW_INTERVAL = 10
# Tolerate this many consecutive Redis errors before dropping leadership.
ERROR_TOLERANCE = 2

Callback = Callable[[], Awaitable[None]]


class LeaderElection:
    def __init__(self) -> None:
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._is_leader = False
        self._on_acquire: list[Callback] = []
        self._on_release: list[Callback] = []
        self._stop = asyncio.Event()
        self._consecutive_errors = 0

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    def on_acquire(self, cb: Callback) -> None:
        self._on_acquire.append(cb)

    def on_release(self, cb: Callback) -> None:
        self._on_release.append(cb)

    async def _set_leader(self, new: bool) -> None:
        if new == self._is_leader:
            return
        self._is_leader = new
        callbacks = self._on_acquire if new else self._on_release
        for cb in callbacks:
            try:
                await cb()
            except Exception as e:
                logger.error(f"Leader {'acquire' if new else 'release'} callback failed: {e}", exc_info=True)

    async def run(self) -> None:
        r = get_redis()
        while not self._stop.is_set():
            try:
                if not self._is_leader:
                    acquired = await r.set(LOCK_KEY, self.instance_id, nx=True, ex=LOCK_TTL)
                    if acquired:
                        logger.info(f"Acquired leader lock as {self.instance_id}")
                        await self._set_leader(True)
                else:
                    # Atomic renew: extend TTL only if we still hold the lock.
                    renewed = await r.set(LOCK_KEY, self.instance_id, xx=True, ex=LOCK_TTL)
                    if not renewed:
                        holder = await r.get(LOCK_KEY)
                        logger.warning(f"Lost leader lock (now held by {holder!r})")
                        await self._set_leader(False)
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(
                    f"Leader election error ({self._consecutive_errors}/{ERROR_TOLERANCE}): {e}",
                    exc_info=True,
                )
                if self._is_leader and self._consecutive_errors >= ERROR_TOLERANCE:
                    logger.warning("Dropping leadership after consecutive Redis errors")
                    await self._set_leader(False)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=RENEW_INTERVAL)
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> None:
        """Release the lock (if held) and stop the loop."""
        self._stop.set()
        if self._is_leader:
            try:
                r = get_redis()
                holder = await r.get(LOCK_KEY)
                if holder == self.instance_id:
                    await r.delete(LOCK_KEY)
                    logger.info("Released leader lock on shutdown")
            except Exception as e:
                logger.error(f"Error releasing leader lock: {e}", exc_info=True)
            await self._set_leader(False)


leader = LeaderElection()
