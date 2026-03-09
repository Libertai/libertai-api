import time

import aiohttp

from src.logger import setup_logger

logger = setup_logger(__name__)

ALEPH_API_URL = (
    "https://api2.aleph.im/api/v0/aggregates/0xe1F7220D201C64871Cefb25320a8a588393eE508.json?keys=LTAI_PRICING"
)


class AlephService:
    __last_fetch_time: float = 0
    __cache_ttl = 300  # 5 minutes
    redirections: dict[str, str] = {}

    async def refresh_redirections(self):
        """Fetch redirections from Aleph and build lookup map."""
        current_time = time.time()
        if (current_time - self.__last_fetch_time) < self.__cache_ttl:
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
                    self.__last_fetch_time = current_time
                    logger.debug(f"Loaded {len(self.redirections)} model redirections")
        except Exception as e:
            logger.error(f"Error fetching Aleph redirections: {e}")

    def resolve(self, model: str) -> str:
        """Return the target model if redirected, else the original."""
        return self.redirections.get(model.lower(), model)


aleph_service = AlephService()
