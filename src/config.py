import os

from dotenv import load_dotenv
from libertai.interfaces.subscription import (
    SubscriptionDefinition,
    SubscriptionProvider,
    SubscriptionType,
)


class _Config:
    ALEPH_API_URL: str | None
    LTAI_SENDER_SK: str
    LTAI_SENDER_ADDRESS: str
    LTAI_AUTH_POST_CHANNEL: str
    LTAI_AUTH_POST_TYPE: bytes
    LTAI_AUTH_METRICS_POST_TYPE: str
    LTAI_AUTH_METRICS_POST_CHANNEL: str
    LTAI_AUTH_REV_TAG: str
    SUBSCRIPTION_POST_SENDER_ADDRESS: str
    SUBSCRIPTION_POST_TYPE: str
    SUBSCRIPTION_POST_CHANNEL: str
    subscription_plans: list[list[SubscriptionDefinition]]

    def __init__(self):
        load_dotenv()

        self.ALEPH_API_URL = os.getenv("ALEPH_API_URL")
        self.LTAI_SENDER_SK = os.getenv("ALEPH_SENDER_SK")
        self.LTAI_SENDER_ADDRESS = os.getenv("ALEPH_SENDER_ADDRESS")
        self.LTAI_AUTH_POST_CHANNEL = os.getenv("LTAI_AUTH_POST_CHANNEL", "libertai-auth")
        self.LTAI_AUTH_POST_TYPE = os.getenv("LTAI_AUTH_POST_TYPE", "libertai-auth-keys")
        self.LTAI_AUTH_METRICS_POST_TYPE = os.getenv("LTAI_AUTH_METRICS_POST_TYPE", "libertai-user-metrics")
        self.LTAI_AUTH_METRICS_POST_CHANNEL = os.getenv("LTAI_AUTH_METRICS_POST_CHANNEL", "libertai-metrics")
        self.LTAI_AUTH_REV_TAG = os.getenv("LTAI_AUTH_REV_TAG", "rev_001")

        self.SUBSCRIPTION_POST_SENDER_ADDRESS = os.getenv("SUBSCRIPTION_POST_SENDER_ADDRESS")
        self.SUBSCRIPTION_POST_TYPE = os.getenv("SUBSCRIPTION_POST_TYPE")
        self.SUBSCRIPTION_POST_CHANNEL = os.getenv("SUBSCRIPTION_POST_CHANNEL")

config = _Config()
