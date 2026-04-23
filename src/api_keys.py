import json

import httpx

from src.config import config
from src.cryptography import create_signed_payload
from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

REDIS_KEY = k("api_keys")


async def get_active_keys() -> set | None:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{config.BACKEND_API_URL}/api-keys/admin/list",
                headers={"x-admin-token": config.BACKEND_SECRET_TOKEN},
            )
            if response.status_code == 200:
                data = response.json()
                return set(data.get("keys") or [])
            logger.error(f"Error fetching accounts: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching accounts {str(e)}", exc_info=True)
        return None


class KeysManager:
    _instance = None
    keys: set[str] = set()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KeysManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def key_exists(self, key):
        return key in self.keys

    async def refresh_keys(self):
        """Leader-only: fetch authoritative keys and publish to Redis."""
        new_keys = await get_active_keys()
        if new_keys is not None:
            self.keys = new_keys
            try:
                await get_redis().set(REDIS_KEY, json.dumps(sorted(new_keys)))
            except Exception as e:
                logger.error(f"Failed to publish keys to Redis: {e}", exc_info=True)
        # Also distribute keys to client servers
        await distribute_keys_to_clients()

    async def sync_from_redis(self):
        """All replicas: refresh local snapshot from Redis.

        A missing key means "leader hasn't published yet" → keep current cache.
        An empty list means "authoritatively empty" → clear local cache.
        """
        try:
            raw = await get_redis().get(REDIS_KEY)
            if raw is None:
                return
            self.keys = set(json.loads(raw))
        except Exception as e:
            logger.error(f"Failed to sync keys from Redis: {e}", exc_info=True)


async def distribute_keys_to_clients():
    """
    Distribute encrypted API keys to all client servers configured in MODELS.
    """
    client_endpoints = set()
    for servers in config.MODELS.values():
        for server in servers:
            client_endpoints.add(f"{server}/libertai/api-keys")

    keys_manager = KeysManager()
    keys_list = list(keys_manager.keys)

    try:
        signed_payload = create_signed_payload({"keys": keys_list}, config.PRIVATE_KEY)
        payload = {"encrypted_payload": signed_payload}

        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in client_endpoints:
                try:
                    response = await client.post(endpoint, json=payload)
                    if response.status_code != 200:
                        logger.error(f"Error sending keys to {endpoint}: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.error(f"Exception sending keys to {endpoint}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error creating signed payload: {e}", exc_info=True)
