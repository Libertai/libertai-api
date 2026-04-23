import json
import time

import httpx

from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

ALEPH_API_URL = (
    "https://api2.aleph.im/api/v0/aggregates/0xe1F7220D201C64871Cefb25320a8a588393eE508.json?keys=LTAI_PRICING"
)

REDIS_KEY = k("aleph", "snapshot")


class AlephService:
    def __init__(self):
        self._last_fetch_time: float = 0
        self._cache_ttl = 300  # 5 minutes
        self.redirections: dict[str, str] = {}
        self.reasoning_models: set[str] = set()

    async def refresh(self):
        """Leader-only: fetch redirections and model capabilities from Aleph and publish to Redis."""
        current_time = time.time()
        if (current_time - self._last_fetch_time) < self._cache_ttl:
            return

        logger.debug("Fetching redirections from Aleph")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(ALEPH_API_URL)
                response.raise_for_status()
                data = response.json()

            pricing_data = data.get("data", {}).get("LTAI_PRICING", {})
            raw_redirections = pricing_data.get("redirections", [])

            new_map = {}
            for r in raw_redirections:
                from_id = r.get("from", "").lower()
                to_id = r.get("to", "").lower()
                if from_id and to_id:
                    new_map[from_id] = to_id

            self.redirections = new_map
            logger.debug(f"Loaded {len(self.redirections)} model redirections")

            raw_models = pricing_data.get("models", [])
            new_reasoning = set()
            for m in raw_models:
                model_id = m.get("id", "").lower()
                reasoning = m.get("capabilities", {}).get("text", {}).get("reasoning", False)
                if model_id and reasoning:
                    new_reasoning.add(model_id)

            self.reasoning_models = new_reasoning
            logger.debug(f"Loaded {len(self.reasoning_models)} reasoning models")

            self._last_fetch_time = current_time

            try:
                await get_redis().set(
                    REDIS_KEY,
                    json.dumps(
                        {
                            "redirections": self.redirections,
                            "reasoning_models": sorted(self.reasoning_models),
                        }
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to publish Aleph snapshot to Redis: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error fetching Aleph data: {e}", exc_info=True)

    async def sync_from_redis(self):
        """All replicas: refresh local snapshot from Redis."""
        try:
            raw = await get_redis().get(REDIS_KEY)
            if raw:
                snap = json.loads(raw)
                self.redirections = dict(snap.get("redirections") or {})
                self.reasoning_models = set(snap.get("reasoning_models") or [])
        except Exception as e:
            logger.error(f"Failed to sync Aleph snapshot from Redis: {e}", exc_info=True)

    def is_reasoning_model(self, model: str) -> bool:
        return model.lower() in self.reasoning_models

    def resolve(self, model: str) -> str:
        """Return the target model if redirected, else the original."""
        return self.redirections.get(model.lower(), model)


aleph_service = AlephService()
