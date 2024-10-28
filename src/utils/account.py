import hashlib
from datetime import datetime

from aleph.sdk import AlephHttpClient, AuthenticatedAlephHttpClient
from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.query.filters import MessageFilter
from aleph_message.models import MessageType
from libertai.crypto.common import decrypt_secret, encrypt_secret, generate_unique_token
from libertai.crypto.ethereum import get_address_from_signature
from libertai.interfaces.subscription import Subscription, SubscriptionProvider
from libertai.utils.signature import get_token_message
from src.config import config
from src.interfaces.account import CreateAccount, TokenAccount


class InvalidSignatureError(Exception):
    "Raised when signature address mismatch"
    pass


class SubscriptionNotFoundError(Exception):
    "Subscription not found"
    pass


class SubscriptionNotValidError(Exception):
    "Subscription not valid"
    pass


async def register_application(account: TokenAccount, token: str):
    aleph_account = ETHAccount(config.LTAI_SENDER_SK)
    encrypted_token = encrypt_secret(aleph_account.get_public_key(), token.encode())
    content = {
        "name": account.name,
        "etk": encrypted_token.hex(),
        "owner": account.owner,
        "tags": [account.owner, config.LTAI_AUTH_REV_TAG]
    }

    async with AuthenticatedAlephHttpClient(
        account=aleph_account, api_server=config.ALEPH_API_URL
    ) as client:
        post_message, _ = await client.create_post(
            post_content=content,
            post_type=config.LTAI_AUTH_POST_TYPE,
            channel=config.LTAI_AUTH_POST_CHANNEL,
            sync=True
        )


def create_token(account):
    token = generate_unique_token()
    return token


async def create_token_from_account(account: CreateAccount):
    message = get_token_message()
    address = get_address_from_signature(message, account.signature)

    if account.account.address != address:
        raise InvalidSignatureError({"message": "message and signature address mismatch!"})

    subscription = await get_subscription(address)
    if subscription is None:
        raise SubscriptionNotFoundError({"message": "Subscription not found!"})

    if not is_subscription_active(subscription):
        raise SubscriptionNotValidError({"message": "Subscription not valid!"})

    token = create_token(account)

    sha1_token = hashlib.sha1(token.encode()).hexdigest()
    account = TokenAccount(
        name="default",
        sha1_token=sha1_token,
        owner=address,
        subscription=subscription
    )

    await register_application(account, token)
    return {
        "account": account,
        "token": token
    }


async def get_active_accounts() -> set:
    accounts = set()
    async with AlephHttpClient(
            api_server=config.ALEPH_API_URL
    ) as client:
        items = await client.get_messages(
            page=1,
            page_size=500,
            message_filter=MessageFilter(
                addresses=[config.LTAI_SENDER_ADDRESS],
                content_types=[config.LTAI_AUTH_POST_TYPE],
                message_types=[MessageType.post],
                channels=[config.LTAI_AUTH_POST_CHANNEL],
                tags=[config.LTAI_AUTH_REV_TAG]
            )
        )
        for item in items.messages:
            data = item.content.content

            subscription = await get_subscription(data["owner"])
            if not is_subscription_active(subscription):
                continue

            token = decrypt_secret(
                config.LTAI_SENDER_SK,
                bytes.fromhex(data["etk"])
            )
            sha1_token = hashlib.sha1(token).hexdigest()

            accounts.add(TokenAccount(
                name=data["name"],
                sha1_token=sha1_token,
                owner=data["owner"],
                subscription=subscription
            ))

    return accounts


async def get_subscription(address: str):
    subscription = None
    async with AlephHttpClient(
            api_server=config.ALEPH_API_URL
    ) as client:
        items = await client.get_messages(
            page=1,
            page_size=1,
            message_filter=MessageFilter(
                addresses=[config.SUBSCRIPTION_POST_SENDER_ADDRESS],
                content_types=[config.SUBSCRIPTION_POST_TYPE],
                message_types=[MessageType.post],
                channels=[config.SUBSCRIPTION_POST_CHANNEL],
                tags=[address, address.lower()]
            )
        )

        if len(items.messages) > 0:
            subscription = Subscription(
                **items.messages[0].content.content
            )

    return subscription


def is_subscription_active(subscription: Subscription):
    if not subscription.is_active:
        return False

    if subscription.ended_at is not None:
        current_unix_timestamp = int(datetime.now().strftime("%s"))

        if subscription.ended_at < current_unix_timestamp:
            return False

    return True
