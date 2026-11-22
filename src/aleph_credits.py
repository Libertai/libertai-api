from http import HTTPStatus

from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.client import AuthenticatedAlephHttpClient
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.config import config
from src.logger import setup_logger
from src.telegram import send_message
from src.x402 import (
    FacilitatorUnavailable,
    accepted_matches,
    decode_payment_header,
    release_payment,
    reserve_payment,
    x402_manager,
)

logger = setup_logger(__name__)

router = APIRouter(prefix="/libertai", tags=["Aleph Credits"])

MAX_CREDIT_AMOUNT = 1000.0  # Maximum single purchase in USD

# Singleton — parsed once at import, reused across requests
_aleph_account: ETHAccount | None = None
if config.ALEPH_SENDER_PRIVATE_KEY:
    try:
        _aleph_account = ETHAccount(private_key=bytes.fromhex(config.ALEPH_SENDER_PRIVATE_KEY.removeprefix("0x")))
    except Exception as e:
        logger.critical(f"Failed to initialize Aleph account — check ALEPH_SENDER_PRIVATE_KEY: {e}")
        raise


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

    if body.amount < config.X402_MIN_PRICE:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Amount must be at least ${config.X402_MIN_PRICE}",
        )

    if body.amount > MAX_CREDIT_AMOUNT:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Amount must not exceed ${MAX_CREDIT_AMOUNT}",
        )

    resource_url = f"{config.PUBLIC_BASE_URL}/libertai/aleph-credits" if config.PUBLIC_BASE_URL else str(request.url)
    description = f"Purchase {body.amount}$ of Aleph credits for {body.address}"

    requirements = await x402_manager.build_requirements("exact", body.amount)
    if requirements is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="x402 payments temporarily unavailable",
        )

    payment_header = request.headers.get("payment-signature")
    if not payment_header:
        return x402_manager.build_402_response(requirements, resource_url, description)

    payment = decode_payment_header(payment_header)
    # Credits delivered equal body.amount, so the settled (signed) amount must equal it too.
    if payment is None or not accepted_matches(payment["accepted"], requirements, exact_amount=True):
        return x402_manager.build_402_response(
            requirements, resource_url, description, error="Payment does not match the requirements"
        )

    try:
        invalid_reason = await x402_manager.verify_payment(payment)
    except FacilitatorUnavailable as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="x402 payments temporarily unavailable",
        )
    if invalid_reason is not None:
        return x402_manager.build_402_response(requirements, resource_url, description, error=invalid_reason)

    try:
        reserved = await reserve_payment(payment)
    except Exception as e:
        logger.error(f"x402 payment reservation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="x402 payments temporarily unavailable",
        )
    if not reserved:
        return x402_manager.build_402_response(requirements, resource_url, description, error="Payment already used")

    # Settle (collect USDC) BEFORE transferring credits.
    settled = await x402_manager.settle_payment(payment)
    if not settled:
        # The facilitator's on-chain nonce check still prevents a double settle if this one landed.
        try:
            await release_payment(payment)
        except Exception as e:
            logger.error(f"x402 payment release failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Payment settlement failed",
        )

    # Payment settled — transfer credits via Aleph SDK
    # Same conversion as the settled amount, so credits delivered equal what was charged.
    credit_amount = int(requirements["amount"])

    try:
        async with AuthenticatedAlephHttpClient(account=_aleph_account) as client:
            message, _status = await client.create_post(
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
        await send_message(
            f"CRITICAL: Payment settled but credit transfer failed\n"
            f"Address: {body.address}\n"
            f"Amount: ${body.amount} ({credit_amount} credits)\n"
            f"Error: {e}"
        )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Credit transfer failed after payment settlement — team has been notified",
        )
