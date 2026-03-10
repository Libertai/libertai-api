import time

import aiohttp

from src.logger import setup_logger

logger = setup_logger(__name__)

ALEPH_API_URL = (
    "https://api2.aleph.im/api/v0/aggregates/0xe1F7220D201C64871Cefb25320a8a588393eE508.json?keys=LTAI_PRICING"
)


class AlephService:
    def __init__(self):
        self._last_fetch_time: float = 0
        self._cache_ttl = 300  # 5 minutes
        self.redirections: dict[str, str] = {}
        self.reasoning_models: set[str] = set()

    async def refresh(self):
        """Fetch redirections and model capabilities from Aleph."""
        current_time = time.time()
        if (current_time - self._last_fetch_time) < self._cache_ttl:
            return

        logger.debug("Fetching redirections from Aleph")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ALEPH_API_URL) as response:
                    response.raise_for_status()
                    data = await response.json()

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
        except Exception as e:
            logger.error(f"Error fetching Aleph data: {e}")

    def is_reasoning_model(self, model: str) -> bool:
        return model.lower() in self.reasoning_models

    def resolve(self, model: str) -> str:
        """Return the target model if redirected, else the original."""
        return self.redirections.get(model.lower(), model)


aleph_service = AlephService()
