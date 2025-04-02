import hashlib
import uuid
from datetime import datetime
from typing import Optional

from libertai_utils.interfaces.subscription import (
    Subscription,
    SubscriptionAccount,
    SubscriptionProvider,
)
from pydantic import BaseModel, PrivateAttr


class CreateAccount(BaseModel):
    account: SubscriptionAccount
    signature: str


class TokenMessage(BaseModel):
    provider: SubscriptionProvider


class TokenAccount(BaseModel):
    name: str
    _token: str = PrivateAttr(default="")
    token: Optional[str] = "**hidden**"
    sha1_token: str
    owner: str
    subscription: Subscription

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __hash__(self) -> str:
        return self.sha1_token.lower().__hash__()

    def set_token(self, value):
        self._token = value

    def get_token(self):
        return self._token

    def reveal_token(self):
        self.token = self.get_token()


class Account(SubscriptionAccount):
    signature: Optional[str]


class AccountListResponse(BaseModel):
    accounts: list[TokenAccount]
    reveal_message: str


class ApiKey(BaseModel):
    def __hash__(self) -> str:
        return self.sha1_token.lower().__hash__()

    @property
    def sha1_token(self):
        return hashlib.sha1(self.key.encode()).hexdigest()

    def owner(self):
        return self.user_address

    id: uuid.UUID
    key: str  # Masked key for display
    name: str
    user_address: str
    created_at: datetime
    is_active: bool
    monthly_limit: float | None = None
