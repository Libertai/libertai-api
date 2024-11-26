from typing import Optional

from libertai_utils.interfaces.subscription import (
    Subscription,
    SubscriptionAccount,
    SubscriptionProvider,
)
from pydantic import BaseModel


class CreateAccount(BaseModel):
    account: SubscriptionAccount
    signature: str


class TokenMessage(BaseModel):
    provider: SubscriptionProvider


class TokenAccount(BaseModel):
    name: str
    sha1_token: str
    owner: str
    subscription: Subscription

    def __hash__(self) -> str:
        return self.sha1_token.lower().__hash__()


class Account(SubscriptionAccount):
    signature: Optional[str]
