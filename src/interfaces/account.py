from typing import Optional

from libertai.interfaces.subscription import (
    SubscriptionAccount,
    SubscriptionProvider,
    SubscriptionType,
)
from pydantic import BaseModel


class CreateAccount(BaseModel):
    account: SubscriptionAccount
    signature: str


class TokenMessage(BaseModel):
    provider: SubscriptionProvider
