import json

import aiohttp

from src.config import config
from src.cryptography import create_signed_payload
from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

REDIS_KEY = k("api_keys")


async def get_active_keys() -> set | None:
    keys = set()

    try:
        async with aiohttp.ClientSession() as session:
            session.headers["x-admin-token"] = config.BACKEND_SECRET_TOKEN
            path = "api-keys/admin/list"
            async with session.get(f"{config.BACKEND_API_URL}/{path}") as response:
                if response.status == 200:
                    data = await response.json()
                    keys.update(data.get("keys"))
                else:
                    logger.error(f"Error fetching accounts: {response.status}")
                    return None

    except Exception as e:
        logger.error(f"Exception fetching accounts {str(e)}", exc_info=True)
        return None

    return keys


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

    # Get all unique server URLs from the models config
    client_endpoints = set()
    for servers in config.MODELS.values():
        for server in servers:
            # Add the libertai endpoint to each server URL
            client_endpoint = f"{server}/libertai/api-keys"
            client_endpoints.add(client_endpoint)

    # Get the current keys
    keys_manager = KeysManager()
    keys_list = list(keys_manager.keys)

    # Create signed payload
    try:
        signed_payload = create_signed_payload({"keys": keys_list}, config.PRIVATE_KEY)
        payload = {"encrypted_payload": signed_payload}

        # Send to all client endpoints
        async with aiohttp.ClientSession() as session:
            for endpoint in client_endpoints:
                try:
                    async with session.post(
                        endpoint, json=payload, headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            await response.json()
                        else:
                            error_text = await response.text()
                            logger.error(f"Error sending keys to {endpoint}: {response.status} - {error_text}")
                except Exception as e:
                    logger.error(f"Exception sending keys to {endpoint}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error creating signed payload: {e}", exc_info=True)
