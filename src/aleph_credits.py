import re
from http import HTTPStatus

from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.client import AuthenticatedAlephHttpClient
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.config import config
from src.logger import setup_logger
from src.telegram import telegram_reporter
from src.x402 import x402_manager

logger = setup_logger(__name__)

router = APIRouter(prefix="/libertai", tags=["Aleph Credits"])

CREDITS_DECIMALS = 6
_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Singleton — parsed once at import, reused across requests
_aleph_account: ETHAccount | None = None
if config.ALEPH_SENDER_PRIVATE_KEY:
    try:
        _aleph_account = ETHAccount(private_key=bytes.fromhex(config.ALEPH_SENDER_PRIVATE_KEY.removeprefix("0x")))
    except Exception as e:
        logger.error(f"Failed to initialize Aleph account — check ALEPH_SENDER_PRIVATE_KEY: {e}")


class AlephCreditsRequest(BaseModel):
    address: str
    amount: float


@router.post("/aleph-credits")
async def purchase_aleph_credits(request: Request, body: AlephCreditsRequest):
    if not _aleph_account:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Aleph credits service not configured",
        )

    if body.amount <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Amount must be positive",
        )

    if not _ETH_ADDRESS_RE.match(body.address):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid Ethereum address",
        )

    # x402 exact payment flow
    # NOTE: thirdweb's /accepts endpoint is deterministic for the same inputs, so
    # re-fetching requirements on the second call (with payment header) will match
    # what the client originally signed against.
    resource_url = f"{config.PUBLIC_BASE_URL}/libertai/aleph-credits" if config.PUBLIC_BASE_URL else str(request.url)

    requirements = await x402_manager.fetch_payment_requirements_exact(
        price=body.amount,
        resource_url=resource_url,
        description=f"Purchase {body.amount}$ of Aleph credits for {body.address}",
    )
    if not requirements or len(requirements) == 0:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Failed to get payment requirements from facilitator",
        )

    payment_header = request.headers.get("x-payment") or request.headers.get("payment-signature")
    if not payment_header:
        return x402_manager.build_402_response(requirements)

    # NOTE: thirdweb's settle endpoint enforces nonce uniqueness, so concurrent
    # requests carrying the same payment will have at most one succeed at settlement.
    valid = await x402_manager.verify_payment(payment_header, requirements[0])
    if not valid:
        return x402_manager.build_402_response(requirements)

    # Settle x402 payment (collect USDC) BEFORE transferring credits
    settled = await x402_manager.settle_payment(payment_header, requirements[0], body.amount)
    if not settled:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Payment settlement failed",
        )

    # Payment settled — transfer credits via Aleph SDK
    credit_amount = int(body.amount * 10**CREDITS_DECIMALS)

    try:
        async with AuthenticatedAlephHttpClient(account=_aleph_account) as client:
            message, status = await client.create_post(
                post_content={
                    "transfer": {
                        "credits": [
                            {
                                "address": body.address,
                                "amount": credit_amount,
                            }
                        ]
                    }
                },
                post_type="aleph_credit_transfer",
                channel="ALEPH_CREDIT",
            )
        logger.info(f"Transferred {credit_amount} credits to {body.address} (message hash: {message.item_hash})")
        return {
            "status": "success",
            "credits_transferred": credit_amount,
            "recipient": body.address,
            "item_hash": message.item_hash,
        }
    except Exception as e:
        # Payment was already settled but credit transfer failed — alert for manual resolution
        logger.error(
            f"CRITICAL: Payment settled but Aleph credit transfer failed for {body.address} "
            f"(amount={body.amount}, credits={credit_amount}): {e}",
            exc_info=True,
        )
        await telegram_reporter.send_message(
            f"CRITICAL: Payment settled but credit transfer failed\n"
            f"Address: {body.address}\n"
            f"Amount: ${body.amount} ({credit_amount} credits)\n"
            f"Error: {e}"
        )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Credit transfer failed after payment settlement — team has been notified",
        )
