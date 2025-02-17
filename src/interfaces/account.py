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
