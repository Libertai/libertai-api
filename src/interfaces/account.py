from typing import Optional

from libertai.interfaces.subscription import (
    SubscriptionAccount,
    SubscriptionProvider,
    SubscriptionType,
)
from pydantic import BaseModel


class CreateAccount(BaseModel):
    id: Optional[str]
    account: SubscriptionAccount
    type: Optional[SubscriptionType]
    provider: SubscriptionProvider
    message: str
    signature: str


class TokenMessage(BaseModel):
    provider: SubscriptionProvider
