import asyncio
import base64
import json
import math
import time
from collections.abc import Collection
from typing import ClassVar, Literal

import httpx
import tiktoken  # type: ignore[import-not-found]
from fastapi import Response
from fastapi.responses import JSONResponse

from src.cdp_auth import CDP_HOST, cdp_auth_headers
from src.config import config
from src.logger import setup_logger
from src.redis_client import get_redis, k

logger = setup_logger(__name__)

REDIS_KEY_PRICES = k("x402", "prices")

# Standard x402 v2 (github.com/coinbase/x402 specs), settled through the CDP facilitator.
X402_VERSION = 2
FACILITATOR_ROUTE = "/platform/v2/x402"
NETWORK = "eip155:8453"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
# EIP-712 domain of USDC on Base: signatures are made over these exact values.
USDC_EIP712_EXTRA = {"name": "USD Coin", "version": "2"}
# Validity window of the payer's signature. upto payments settle after the
# response finishes (via the usage report), so this must outlast the longest request.
MAX_TIMEOUT_SECONDS = 3600
SUPPORTED_TTL_SECONDS = 3600
# Tolerance for payer clock skew when checking that a signature outlives settlement.
VALIDITY_SKEW_SECONDS = 300

Scheme = Literal["exact", "upto"]


class FacilitatorUnavailable(Exception):
    """The facilitator could not be reached or failed: the payment's validity is unknown."""


_enc = tiktoken.get_encoding("cl100k_base")

# These endpoints need no forward proxy: a module-level client reuses
# connections instead of paying a fresh TCP+TLS handshake per x402 request.
limits = httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=300.0)
client = httpx.AsyncClient(timeout=30.0, limits=limits)


async def close_http_client() -> None:
    await client.aclose()


def usd_to_atomic(usd: float) -> str:
    """USD → USDC atomic units (6 decimals), rounded up so a quote never undercharges."""
    return str(max(1, math.ceil(round(usd * 1_000_000, 6))))


def decode_payment_header(header: str) -> dict | None:
    """Decode a v2 PAYMENT-SIGNATURE header (base64 JSON PaymentPayload); None if malformed."""
    try:
        payment = json.loads(base64.b64decode(header, validate=True))
    except ValueError:
        return None
    if (
        not isinstance(payment, dict)
        or payment.get("x402Version") != X402_VERSION
        or not isinstance(payment.get("accepted"), dict)
        or not isinstance(payment.get("payload"), dict)
    ):
        return None
    return payment


def accepted_matches(
    accepted: dict, required: dict, exact_amount: bool = False, facilitators: Collection[str] = ()
) -> bool:
    """Whether the requirements the payer signed for are acceptable for this request.

    By default not an exact match: prices can refresh between the 402 and the paid
    retry, so paying at least the current price is enough. `exact_amount` is for
    endpoints that deliver exactly what is paid. `facilitators` (lowercase) are the
    facilitator's signers: any of them can settle, and each /supported call (so each
    replica's quote) may name a different one. Everything else that decides where the
    money goes must match exactly.
    """
    try:
        signed, quoted = int(accepted.get("amount", "")), int(required["amount"])
        amount_ok = signed == quoted if exact_amount else signed >= quoted
    except (TypeError, ValueError):
        return False

    def same_address(a: object, b: object) -> bool:
        return isinstance(a, str) and isinstance(b, str) and a.lower() == b.lower()

    raw_extra = accepted.get("extra")
    accepted_extra: dict = raw_extra if isinstance(raw_extra, dict) else {}
    try:
        # The payer derives its signature's expiry from this; shorter than ours can expire before settlement.
        timeout_ok = int(accepted.get("maxTimeoutSeconds", 0)) >= int(required["maxTimeoutSeconds"])
    except (TypeError, ValueError):
        return False
    required_extra = required["extra"]
    return (
        amount_ok
        and timeout_ok
        and accepted.get("scheme") == required["scheme"]
        and accepted.get("network") == required["network"]
        and same_address(accepted.get("asset"), required["asset"])
        and same_address(accepted.get("payTo"), required["payTo"])
        and all(
            (
                same_address(accepted_extra.get(key), value)
                or (isinstance(accepted_extra.get(key), str) and accepted_extra[key].lower() in facilitators)
            )
            if key == "facilitatorAddress"
            else accepted_extra.get(key) == value
            for key, value in required_extra.items()
        )
    )


def _authorization(payment: dict) -> dict | None:
    """The signed transfer terms for the accepted scheme: EIP-3009 `authorization` (exact) or
    `permit2Authorization` (upto). None if the payload carries the other one too: the
    facilitator reads one by scheme, so a decoy in the other could steer our own checks."""
    payload = payment["payload"]
    field, other = (
        ("permit2Authorization", "authorization")
        if payment["accepted"].get("scheme") == "upto"
        else ("authorization", "permit2Authorization")
    )
    authorization = payload.get(field)
    if other in payload or not isinstance(authorization, dict):
        return None
    return authorization


def payment_valid_until(payment: dict) -> int | None:
    """Expiry (unix seconds) of the signed authorization: EIP-3009 validBefore or Permit2 deadline."""
    authorization = _authorization(payment)
    if authorization is None:
        return None
    try:
        # Only the expiry field the facilitator checks for this scheme counts.
        field = "deadline" if payment["accepted"].get("scheme") == "upto" else "validBefore"
        return int(authorization[field])
    except (KeyError, TypeError, ValueError):
        return None


def outlives_settlement(payment: dict) -> bool:
    """Whether the authorization stays valid long enough to be settled after the request completes."""
    valid_until = payment_valid_until(payment)
    return valid_until is not None and valid_until >= time.time() + MAX_TIMEOUT_SECONDS - VALIDITY_SKEW_SECONDS


def _reservation_key(payment: dict) -> str | None:
    """Redis key identifying a signed payment by what the signature commits to; None if not identifiable."""
    authorization = _authorization(payment)
    if authorization is None:
        return None
    payer = authorization.get("from")
    # The signature commits to the nonce's numeric value (bytes32 for EIP-3009, uint256 for
    # Permit2), not its spelling: key on the value so hex case or zero padding can't vary it.
    try:
        nonce = int(str(authorization.get("nonce")), 0)
    except ValueError:
        return None
    if not isinstance(payer, str):
        return None
    accepted = payment["accepted"]
    return k(
        "x402", "used", str(accepted.get("network")), str(accepted.get("asset")).lower(), payer.lower(), str(nonce)
    )


async def reserve_payment(payment: dict) -> bool:
    """Claim a signed payment for one request, across every x402 endpoint; False if already used.

    Settlement can happen after the response, so without this one signature could be
    replayed on concurrent requests that all pass /verify before any settles. The key
    is what the signature commits to (payer + nonce), so unsigned payload fields can't
    vary it; it is held until the authorization expires.
    """
    key = _reservation_key(payment)
    valid_until = payment_valid_until(payment)
    if key is None or valid_until is None:
        return False
    ttl = max(valid_until - int(time.time()), 0) + VALIDITY_SKEW_SECONDS
    return bool(await get_redis().set(key, 1, nx=True, ex=ttl))


def _as_count(value: object) -> int:
    """A choice count as a backend would coerce it (int, integral float or numeric string); 1 if unusable."""
    if isinstance(value, bool):
        return 1
    try:
        count = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return int(count) if count.is_integer() and count > 1 else 1


async def release_payment(payment: dict) -> None:
    """Undo reserve_payment for a payment known not to have been settled, so it can be retried."""
    key = _reservation_key(payment)
    if key is not None:
        await get_redis().delete(key)


class X402Manager:
    _instance = None
    prices: ClassVar[dict[str, dict]] = {}
    _upto_kind_extra: ClassVar[dict | None] = None
    _upto_kind_fetched_at: ClassVar[float] = 0.0
    _evm_signers: ClassVar[frozenset[str]] = frozenset()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def get_price_info(self, model: str) -> dict | None:
        return self.prices.get(model)

    async def compute_price(self, model: str, body: dict) -> tuple[float, Scheme] | None:
        """USD price of a request and the scheme to charge it with.

        Image, audio and embedding prices depend only on the request, so they are
        charged `exact`. Text output length is unknown upfront: `upto` authorizes a
        maximum (input + max_tokens) and settles the actual usage later.
        Every price is floored at X402_MIN_PRICE.
        """
        info = self.get_price_info(model)
        if not info:
            return None

        price: float
        scheme: Scheme = "exact"
        if "price_per_image" in info:
            price = info["price_per_image"] * _as_count(body.get("n"))
        # Audio (TTS) models are billed on the raw input character count (NOT tokens).
        elif info.get("is_audio"):
            inputs = body.get("input", "")
            input_text = inputs if isinstance(inputs, str) else json.dumps(inputs)
            price = len(input_text) / 1_000_000 * info["price_per_million_input_characters"]
        # Embedding models are input-only: price the `input` payload, no completion tokens.
        elif info.get("is_embedding"):
            inputs = body.get("input", "")
            input_text = inputs if isinstance(inputs, str) else json.dumps(inputs)
            input_tokens = await asyncio.to_thread(lambda: len(_enc.encode(input_text)))
            price = input_tokens / 1_000_000 * info["price_per_million_input_tokens"]
        else:
            scheme = "upto"
            # The whole body: the backend may turn any field (messages, prompt, tools, ...) into
            # input tokens, and over-counting only raises the authorized maximum, not the charge.
            input_text = json.dumps(body)
            input_tokens = await asyncio.to_thread(lambda: len(_enc.encode(input_text)))
            # Backends honor any of these fields and may coerce non-int values, so price on the
            # largest, and on the model maximum unless every given limit is a positive int.
            limits = [
                body[f]
                for f in ("max_tokens", "max_completion_tokens", "max_output_tokens")
                if body.get(f) is not None
            ]
            if limits and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in limits):
                max_tokens = max(limits)
            else:
                max_tokens = info.get("default_max_tokens", 4096)
            # Each of n / best_of choices can produce up to max_tokens.
            max_tokens *= max(1, *(_as_count(body.get(field)) for field in ("n", "best_of")))
            price = (
                input_tokens / 1_000_000 * info["price_per_million_input_tokens"]
                + max_tokens / 1_000_000 * info["price_per_million_output_tokens"]
            )
        return max(price, config.X402_MIN_PRICE), scheme

    @staticmethod
    async def _facilitator_post(endpoint: str, body: dict, timeout: float = 30.0) -> httpx.Response:
        path = f"{FACILITATOR_ROUTE}/{endpoint}"
        headers = cdp_auth_headers(config.CDP_API_KEY_ID, config.CDP_API_KEY_SECRET, "POST", path)
        return await client.post(f"https://{CDP_HOST}{path}", json=body, headers=headers, timeout=timeout)

    async def get_upto_kind_extra(self) -> dict | None:
        """`extra` of the facilitator's upto kind on Base, from its /supported.

        It holds `facilitatorAddress`: upto payers sign a Permit2 witness naming this
        address, and only it can settle.
        """
        cls = type(self)
        if cls._upto_kind_extra and time.monotonic() - cls._upto_kind_fetched_at < SUPPORTED_TTL_SECONDS:
            return cls._upto_kind_extra
        path = f"{FACILITATOR_ROUTE}/supported"
        try:
            headers = cdp_auth_headers(config.CDP_API_KEY_ID, config.CDP_API_KEY_SECRET, "GET", path)
            response = await client.get(f"https://{CDP_HOST}{path}", headers=headers)
            response.raise_for_status()
            data = response.json()
            signers = data.get("signers") or {}
            evm_signers = [*signers.get("eip155:*", []), *signers.get(NETWORK, [])]
            cls._evm_signers = frozenset(a.lower() for a in evm_signers if isinstance(a, str))
            for kind in data.get("kinds", []):
                extra = kind.get("extra")
                if (
                    kind.get("x402Version") == X402_VERSION
                    and kind.get("scheme") == "upto"
                    and kind.get("network") == NETWORK
                    and isinstance(extra, dict)
                    and extra.get("facilitatorAddress")
                ):
                    cls._upto_kind_extra = extra
                    cls._upto_kind_fetched_at = time.monotonic()
                    return extra
            logger.error(f"CDP /supported lists no upto kind on {NETWORK}")
        except Exception as e:
            logger.error(f"CDP /supported failed: {e}", exc_info=True)
        # A stale value beats refusing payments while CDP is unreachable.
        return cls._upto_kind_extra

    @property
    def facilitator_signers(self) -> frozenset[str]:
        """Lowercase EVM addresses the facilitator settles from, as of the last /supported."""
        return type(self)._evm_signers

    async def build_requirements(self, scheme: Scheme, price: float) -> dict | None:
        """v2 PaymentRequirements for one request; None if the upto facilitator is unknown."""
        extra: dict = dict(USDC_EIP712_EXTRA)
        if scheme == "upto":
            kind_extra = await self.get_upto_kind_extra()
            if not kind_extra:
                return None
            # Mirrors the facilitator's kind extra so any field it requires for signing is forwarded.
            extra = {**extra, "assetTransferMethod": "permit2", **kind_extra}
        return {
            "scheme": scheme,
            "network": NETWORK,
            "amount": usd_to_atomic(price),
            "asset": USDC_BASE_ADDRESS,
            "payTo": config.X402_WALLET_ADDRESS,
            "maxTimeoutSeconds": MAX_TIMEOUT_SECONDS,
            "extra": extra,
        }

    @staticmethod
    def build_402_response(
        requirements: dict, resource_url: str, description: str, error: str = "PAYMENT-SIGNATURE header is required"
    ) -> Response:
        """402 carrying the v2 PaymentRequired object, in the PAYMENT-REQUIRED header and the body."""
        payment_required = {
            "x402Version": X402_VERSION,
            "error": error,
            "resource": {"url": resource_url, "description": description, "mimeType": "application/json"},
            "accepts": [requirements],
        }
        encoded = base64.b64encode(json.dumps(payment_required).encode()).decode()
        return JSONResponse(status_code=402, content=payment_required, headers={"PAYMENT-REQUIRED": encoded})

    @classmethod
    async def verify_payment(cls, payment: dict) -> str | None:
        """Verify a decoded payment against the requirements it signed; None if valid, else the reason.

        Raises FacilitatorUnavailable when the facilitator itself fails, so the payment
        is not reported as invalid.
        """
        try:
            response = await cls._facilitator_post(
                "verify",
                {"x402Version": X402_VERSION, "paymentPayload": payment, "paymentRequirements": payment["accepted"]},
            )
            data = response.json()
        except Exception as e:
            raise FacilitatorUnavailable(f"CDP verify failed: {e}") from e
        # Only 200/400 judge the payment; anything else (auth, routing, rate limit, 5xx) is on our side.
        if response.status_code not in (200, 400) or not isinstance(data, dict):
            raise FacilitatorUnavailable(f"CDP verify error: {response.status_code} - {response.text}")
        if response.status_code == 200 and data.get("isValid"):
            return None
        logger.warning(f"CDP verify rejected payment: {response.status_code} - {response.text}")
        return data.get("invalidReason") or "invalid_payment"

    @classmethod
    async def settle_payment(cls, payment: dict) -> bool:
        """Settle an exact payment for the amount it signed."""
        try:
            response = await cls._facilitator_post(
                "settle",
                {"x402Version": X402_VERSION, "paymentPayload": payment, "paymentRequirements": payment["accepted"]},
                # On-chain confirmation can take a while.
                timeout=120.0,
            )
            data = response.json()
            if response.status_code == 200 and data.get("success"):
                logger.info(f"x402 payment settled: {data.get('transaction')}")
                return True
            logger.error(f"CDP settle error: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"CDP settle failed: {e}", exc_info=True)
            return False

    async def refresh_prices(self):
        """Leader-only: pull per-token prices from backend and publish to Redis."""
        try:
            response = await client.get(
                f"{config.BACKEND_API_URL}/x402/prices",
                headers={"x-admin-token": config.BACKEND_SECRET_TOKEN},
            )
            if response.status_code == 200:
                self.prices = response.json()
                logger.debug(f"Refreshed x402 prices: {len(self.prices)} models")
                try:
                    await get_redis().set(REDIS_KEY_PRICES, json.dumps(self.prices))
                except Exception as e:
                    logger.error(f"Failed to publish x402 prices to Redis: {e}", exc_info=True)
            else:
                logger.error(f"Error fetching x402 prices: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception fetching x402 prices: {e}", exc_info=True)

    async def sync_from_redis(self):
        """All replicas: refresh local snapshot from Redis."""
        try:
            raw = await get_redis().get(REDIS_KEY_PRICES)
            if raw:
                self.prices = json.loads(raw)
        except Exception as e:
            logger.error(f"Failed to sync x402 prices from Redis: {e}", exc_info=True)


x402_manager = X402Manager()
