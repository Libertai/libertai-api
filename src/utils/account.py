import hashlib

from aleph.sdk import AlephHttpClient, AuthenticatedAlephHttpClient
from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.query.filters import MessageFilter
from aleph_message.models import MessageType
from libertai.crypto.common import decrypt_secret, encrypt_secret, generate_unique_token
from libertai.crypto.ethereum import get_address_from_signature
from libertai.interfaces.subscription import SubscriptionProvider
from libertai.utils.signature import get_token_message
from src.config import config
from src.interfaces.account import CreateAccount


class InvalidSignatureError(Exception):
    "Raised when signature address mismatch"
    pass


async def register_app(token_info: dict):
    aleph_account = ETHAccount(config.LTAI_SENDER_SK)
    encrypted_token = encrypt_secret(aleph_account.get_public_key(), token_info["token"].encode())
    content = {
        "name": token_info["name"],
        "etk": encrypted_token.hex(),
        "tags": [token_info["owner"]]
    }

    async with AuthenticatedAlephHttpClient(
        account=aleph_account, api_server=config.ALEPH_API_URL
    ) as client:
        post_message, _ = await client.create_post(
            post_content=content,
            post_type=config.LTAI_AUTH_POST_TYPE,
            channel=config.LTAI_AUTH_POST_CHANNEL,
        )


def create_token(account):
    token = generate_unique_token()
    return token

async def create_token_from_account(account: CreateAccount):
    message = get_token_message()
    address = get_address_from_signature(message, account.signature)

    if account.account.address != address:
        raise InvalidSignatureError({"message": "message and signature address mismatch!"})

    token = create_token(account)

    token_info = {
        "owner": address,
        "name": "default",
        "token": token,
    }
    await register_app(token_info)
    return token_info


async def get_active_tokens() -> set:
    tokens = set()
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
                channels=[config.LTAI_AUTH_POST_CHANNEL]
            )
        )
        for item in items.messages:
            token = decrypt_secret(
                config.LTAI_SENDER_SK,
                bytes.fromhex(item.content.content["etk"])
                )

            sha1_token = hashlib.sha1(token).hexdigest()
            tokens.add(sha1_token)

    return tokens
