from src.interfaces.metrics import Metric, Metrics
from src.utils.account import get_active_accounts
from src.utils.common import current_timestamp_5min_interval


class AccountManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AccountManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Check if already initialized
            self.tokens = set()
            """Subscription metadata"""
            self.accounts = {}
            """Save request calls"""
            self.calls = {}

            self.initialized = True  # Flag to prevent re-initialization

    def add_accounts(self, accounts):
        tokens = set(map(lambda x: x.sha1_token, accounts))
        self.tokens.update(tokens)
        [self.add_account(account) for account in accounts]

        print(self.tokens)
        print(self.accounts)

    def add_account(self, account):
        self.tokens.add(account.sha1_token)
        if account.sha1_token == "98ba756675cc9ce059b669b16e1cbfd4e01799bf":
            self.increment_calls(account.sha1_token)
            self.increment_calls(account.sha1_token)
        self.accounts[account.sha1_token] = account

    def token_exists(self, sha1_token):
        return sha1_token in self.tokens

    def get_account(self, sha1_token):
        if sha1_token in self.accounts:
            return self.accounts[sha1_token]
        else:
            return None

    def update_metadata(self, token, metadata):
        if token in self.tokens:
            self.metadata[token] = metadata
        else:
            #raise ValueError("Token does not exist")
            pass

    async def load_metadata(self):
        for token in self.tokens:
            print(f"fetch metadata for token: {token}")

    def get_calls(self, sha1_token, since=int):
        return self.calls.get(sha1_token)  # Returns None if token not found

    def update_calls(self, key, calls: int):
        self.calls[key] = calls

    def increment_calls(self, token):
        current_unix_timestamp = current_timestamp_5min_interval().strftime("%s")
        key = f"{token}.{current_unix_timestamp}"
        calls = self.get_calls(key)

        if calls:
            calls += 1
        else:
            calls = 1

        self.update_calls(key, calls)

    def get_metrics(self, token=None, owner=None) -> Metrics:
        metrics = []

        for item, count in self.calls.items():
            key, ts = list(item.split("."))
            if token is not None and token != key:
                continue

            account = self.get_account(key)
            if account is None:
                continue

            if owner is not None and owner != account.owner:
                continue

            metrics.append({
                "owner": account.owner,
                "key": key,
                "ts": ts,
                "count": count
            })

        if metrics:
            return Metrics(values=metrics, owner=owner)
        else:
            return None

    async def load_active_accounts(self):
        accounts = await get_active_accounts()
        self.add_accounts(accounts)
        print(f"Loaded {len(accounts)} accounts")
