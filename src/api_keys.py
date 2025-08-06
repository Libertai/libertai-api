import aiohttp

from src.config import config
from src.cryptography import create_signed_payload


async def get_active_keys() -> set:
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
                    print(f"Error fetching accounts: {response.status}")
                    return keys

    except Exception as e:
        print(f"Exception fetching accounts {str(e)}")
        return keys

    return keys


class KeysManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KeysManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):  # Check if already initialized
            self.keys = set()

    def add_keys(self, keys):
        self.keys.update(keys)

    def key_exists(self, key):
        return key in self.keys

    async def refresh_keys(self):
        self.keys = await get_active_keys()
        # Also distribute keys to client servers
        await distribute_keys_to_clients()


async def distribute_keys_to_clients():
    """
    Distribute encrypted API keys to all client servers configured in MODELS.
    """

    # Get all unique server URLs from the models config
    client_endpoints = set()
    for servers in config.MODELS.values():
        for server in servers:
            # Add the libertai endpoint to each server URL
            client_endpoint = f"{server.url}/libertai/api-keys"
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
                            # TODO: use logger and enable this when all backends are ready
                            # error_text = await response.text()
                            # print(f"Error sending keys to {endpoint}: {response.status} - {error_text}")
                            pass
                except Exception as e:
                    print(f"Exception sending keys to {endpoint}: {e}")

    except Exception as e:
        print(f"Error creating signed payload: {e}")
