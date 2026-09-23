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

# Aleph sits behind no proxy; a module-level client reuses connections instead
# of paying a fresh TCP+TLS handshake per fetch. keepalive_expiry outlasts the
# 300s cache TTL so a connection survives between refresh fetches.
client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(keepalive_expiry=310.0))


async def close_http_client() -> None:
    await client.aclose()


class AlephService:
    def __init__(self):
        self._last_fetch_time: float = 0
        self._cache_ttl = 300  # 5 minutes
        self.redirections: dict[str, str] = {}
        self.reasoning_models: set[str] = set()
        self.vision_models: set[str] = set()
        self.models: dict[str, dict] = {}
        # Distinguishes "no snapshot yet" (serve 503) from "authoritatively empty".
        self.models_loaded = False

    async def refresh(self):
        """Leader-only: fetch redirections and model capabilities from Aleph and publish to Redis."""
        current_time = time.time()
        if (current_time - self._last_fetch_time) < self._cache_ttl:
            return

        logger.debug("Fetching redirections from Aleph")
        try:
            response = await client.get(ALEPH_API_URL)
            response.raise_for_status()
            data = response.json()

            pricing_data = data.get("data", {}).get("LTAI_PRICING")
            raw_models = pricing_data.get("models") if isinstance(pricing_data, dict) else None
            if not isinstance(raw_models, list):
                # Missing or malformed payload (API error, schema change): keep the
                # previous snapshot and retry next cycle instead of treating the
                # models list as authoritatively empty.
                logger.error("Aleph response has no valid LTAI_PRICING.models list; keeping the previous snapshot")
                return

            raw_redirections = pricing_data.get("redirections", [])

            new_map = {}
            for r in raw_redirections:
                from_id = r.get("from", "").lower()
                to_id = r.get("to", "").lower()
                if from_id and to_id:
                    new_map[from_id] = to_id

            self.redirections = new_map
            logger.debug(f"Loaded {len(self.redirections)} model redirections")

            new_reasoning = set()
            new_vision = set()
            new_models = {}
            for m in raw_models:
                model_id = m.get("id", "").lower()
                if not model_id:
                    continue
                new_models[model_id] = m
                text_caps = m.get("capabilities", {}).get("text", {})
                if text_caps.get("reasoning", False):
                    new_reasoning.add(model_id)
                if text_caps.get("vision", False):
                    new_vision.add(model_id)

            if self.models and not new_models:
                logger.warning("Aleph aggregate now reports zero models; publishing an authoritative empty snapshot")

            self.reasoning_models = new_reasoning
            self.vision_models = new_vision
            self.models = new_models
            self.models_loaded = True
            logger.debug(
                f"Loaded {len(self.reasoning_models)} reasoning models, {len(self.vision_models)} vision models"
            )

            self._last_fetch_time = current_time

            try:
                await get_redis().set(
                    REDIS_KEY,
                    json.dumps(
                        {
                            "redirections": self.redirections,
                            "reasoning_models": sorted(self.reasoning_models),
                            "vision_models": sorted(self.vision_models),
                            "models": self.models,
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
                self.vision_models = set(snap.get("vision_models") or [])
                # A snapshot from a pre-"models" release must not clear the local
                # cache or mark metadata as loaded.
                if "models" in snap:
                    self.models = dict(snap.get("models") or {})
                    self.models_loaded = True
        except Exception as e:
            logger.error(f"Failed to sync Aleph snapshot from Redis: {e}", exc_info=True)

    def is_reasoning_model(self, model: str) -> bool:
        return model.lower() in self.reasoning_models

    def is_vision_model(self, model: str) -> bool:
        return model.lower() in self.vision_models

    def get_model(self, model: str) -> dict | None:
        """Return the aggregate metadata entry for a model, if it is priced on Aleph."""
        return self.models.get(model.lower())

    def resolve(self, model: str) -> str:
        """Return the target model if redirected, else the original."""
        return self.redirections.get(model.lower(), model)


aleph_service = AlephService()
