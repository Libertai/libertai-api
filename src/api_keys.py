import json

import httpx

from src.config import config
from src.cryptography import create_signed_payload
from src.logger import setup_logger
from src.redis_client import get_redis, k
from src.ssl_trust import SSL_CONTEXT

logger = setup_logger(__name__)

# Legacy list-shaped snapshot; still written so pre-invalid-map replicas can read it
# during a rolling deploy. Drop once the fleet only runs dict-shape-aware code.
REDIS_KEY = k("api_keys")
# Dict shape: {"keys": [...], "invalid_keys": {key: {"reason", "message"}}}
REDIS_KEY_V2 = k("api_keys_v2")


async def get_active_keys() -> tuple[set, dict] | None:
    try:
        async with httpx.AsyncClient(timeout=120.0, verify=SSL_CONTEXT) as client:
            response = await client.get(
                f"{config.BACKEND_API_URL}/api-keys/admin/list",
                headers={"x-admin-token": config.BACKEND_SECRET_TOKEN},
            )
            if response.status_code == 200:
                data = response.json()
                # invalid_keys entries ({reason, message}) are trusted server-side data,
                # stored/served as-is; consumers read them with .get() fallbacks.
                return set(data.get("keys") or []), dict(data.get("invalid_keys") or {})
            logger.error(f"Error fetching accounts: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching accounts {str(e)}", exc_info=True)
        return None


def parse_snapshot(raw: str) -> tuple[set[str], dict[str, dict]]:
    """Accepts both snapshot shapes: legacy JSON list and v2 dict."""
    data = json.loads(raw)
    if isinstance(data, dict):
        return set(data.get("keys") or []), dict(data.get("invalid_keys") or {})
    return set(data), {}


class KeysManager:
    _instance = None
    keys: set[str] = set()
    # key -> {"reason": str, "message": str} for real-but-unusable keys (limits/credits/disabled)
    invalid_keys: dict[str, dict] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KeysManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def key_exists(self, key):
        return key in self.keys

    def key_invalid_info(self, key: str) -> dict | None:
        return self.invalid_keys.get(key)

    async def refresh_keys(self):
        """Leader-only: fetch authoritative keys and publish to Redis."""
        fetched = await get_active_keys()
        if fetched is not None:
            new_keys, new_invalid = fetched
            self.keys = new_keys
            self.invalid_keys = new_invalid
            try:
                redis = get_redis()
                async with redis.pipeline(transaction=True) as pipe:
                    pipe.set(REDIS_KEY, json.dumps(sorted(new_keys)))
                    pipe.set(
                        REDIS_KEY_V2,
                        json.dumps({"keys": sorted(new_keys), "invalid_keys": new_invalid}),
                    )
                    await pipe.execute()
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
            raw = await get_redis().get(REDIS_KEY_V2)
            if raw is None:
                raw = await get_redis().get(REDIS_KEY)
            if raw is None:
                return
            self.keys, self.invalid_keys = parse_snapshot(raw)
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
        # Old boxes read only "keys" from the decrypted payload; extra fields are ignored.
        signed_payload = create_signed_payload(
            {"keys": keys_list, "invalid_keys": keys_manager.invalid_keys}, config.PRIVATE_KEY
        )
        payload = {"encrypted_payload": signed_payload}

        async with httpx.AsyncClient(timeout=30.0, verify=SSL_CONTEXT) as client:
            for endpoint in client_endpoints:
                try:
                    response = await client.post(endpoint, json=payload)
                    if response.status_code != 200:
                        logger.error(f"Error sending keys to {endpoint}: {response.status_code} - {response.text}")
                except (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException, httpx.ProxyError) as e:
                    # Transient: upstream box slow/unreachable — other endpoints still get their keys
                    logger.warning(f"Could not send keys to {endpoint}: {type(e).__name__}: {e}")
                except Exception as e:
                    logger.error(f"Exception sending keys to {endpoint}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error creating signed payload: {e}", exc_info=True)
