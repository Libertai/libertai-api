from src.utils.account import get_active_tokens
from src.utils.common import current_timestamp_5min_interval


class TokenManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TokenManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Check if already initialized
            self.tokens = set()
            """Subscription metadata"""
            self.metadata = {}
            """Save request calls"""
            self.requests = {}

            self.initialized = True  # Flag to prevent re-initialization

    def add_tokens(self, tokens):
        self.tokens.update(tokens)

    def add_token(self, token, metadata=None):
        self.tokens.add(token)
        if metadata is not None:
            self.metadata[token] = metadata

    def exists(self, token):
        return token in self.tokens

    def get_metadata(self, token):
        return self.metadata.get(token)  # Returns None if token not found

    def update_metadata(self, token, metadata):
        if token in self.tokens:
            self.metadata[token] = metadata
        else:
            #raise ValueError("Token does not exist")
            pass

    async def load_metadata(self):
        for token in self.tokens:
            print(f"fetch metadata for token: {token}")

    def get_requests(self, sha1_token, since=int):
        return self.requests.get(sha1_token)  # Returns None if token not found

    def update_requests(self, key, calls: int):
        self.requests[key] = calls

    def increment_requests(self, token):
        current_unix_timestamp = current_timestamp_5min_interval().strftime("%s")
        key = f"{token}.{current_unix_timestamp}"
        calls = self.get_requests(key)

        if calls:
            calls += 1
        else:
            calls = 1

        self.update_requests(key, calls)

    def get_metrics(self, token=None):
        metrics = {}

        for item, value in self.requests.items():
            key, ts = list(item.split("."))
            if token is not None and token != key:
                continue

            if key in metrics:
                metrics[key][ts] = value
            else:
                metrics[key] = {ts: value}

        return metrics

    async def load_active_tokens(self):
        tokens = await get_active_tokens()
        self.add_tokens(tokens)
        print(f"Loaded {len(tokens)} tokens")
        print("Load metadata...")
        await self.load_metadata()
