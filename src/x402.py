import asyncio
import base64
import json

import aiohttp
import tiktoken  # type: ignore[import-not-found]
from fastapi import Response
from fastapi.responses import JSONResponse

from src.config import config
from src.logger import setup_logger

logger = setup_logger(__name__)

THIRDWEB_X402_BASE = "https://api.thirdweb.com/v1/payments/x402"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

_enc = tiktoken.get_encoding("cl100k_base")


class X402Manager:
    _instance = None
    prices: dict[str, dict] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def get_price_info(self, model: str) -> dict | None:
        return self.prices.get(model)

    async def compute_max_price(self, model: str, body: dict) -> float | None:
        """Compute max price based on input tokens + max_tokens."""
        info = self.get_price_info(model)
        if not info:
            return None

        if "price_per_image" in info:
            return info["price_per_image"]

        messages = body.get("messages", [])
        messages_text = json.dumps(messages)
        input_tokens = await asyncio.to_thread(lambda: len(_enc.encode(messages_text)))

        max_tokens = (
            body.get("max_tokens") or body.get("max_completion_tokens") or info.get("default_max_tokens", 4096)
        )

        price = (
            input_tokens / 1_000_000 * info["price_per_million_input_tokens"]
            + max_tokens / 1_000_000 * info["price_per_million_output_tokens"]
        )
        return max(price, 0.0001)

    @staticmethod
    async def _fetch_requirements(payload: dict) -> list[dict] | None:
        """Fetch payment requirements from thirdweb /accepts endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{THIRDWEB_X402_BASE}/accepts",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-secret-key": config.THIRDWEB_SECRET_KEY,
                    },
                ) as response:
                    if response.status == 402:
                        data = await response.json()
                        return data.get("accepts", [])
                    else:
                        error_text = await response.text()
                        logger.error(f"thirdweb /accepts error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"thirdweb /accepts exception: {e}", exc_info=True)
            return None

    @staticmethod
    async def fetch_payment_requirements(model: str, max_price: float, resource_url: str) -> list[dict] | None:
        """Fetch upTo payment requirements for inference."""
        return await X402Manager._fetch_requirements(
            {
                "resourceUrl": resource_url,
                "method": "POST",
                "network": "eip155:8453",
                "price": {
                    "amount": str(int(max_price * 1_000_000)),
                    "asset": {
                        "address": USDC_BASE_ADDRESS,
                        "decimals": 6,
                    },
                },
                "scheme": "upto",
                "serverWalletAddress": config.X402_SERVER_WALLET_ADDRESS,
                "recipientAddress": config.X402_WALLET_ADDRESS,
                "x402Version": 2,
                "routeConfig": {
                    "description": f"Pay-per-use inference for {model}",
                    "mimeType": "application/json",
                },
            }
        )

    @staticmethod
    async def fetch_payment_requirements_exact(price: float, resource_url: str, description: str) -> list[dict] | None:
        """Fetch exact payment requirements for a fixed-price endpoint."""
        return await X402Manager._fetch_requirements(
            {
                "resourceUrl": resource_url,
                "method": "POST",
                "network": "eip155:8453",
                "price": {
                    "amount": str(int(price * 1_000_000)),
                    "asset": {
                        "address": USDC_BASE_ADDRESS,
                        "decimals": 6,
                    },
                },
                "scheme": "exact",
                "serverWalletAddress": config.X402_SERVER_WALLET_ADDRESS,
                "recipientAddress": config.X402_WALLET_ADDRESS,
                "x402Version": 2,
                "routeConfig": {
                    "description": description,
                    "mimeType": "application/json",
                },
            }
        )

    @staticmethod
    def build_402_response(requirements: list[dict]) -> Response:
        """Build 402 response with requirements from thirdweb."""
        payment_required = {
            "x402Version": 2,
            "accepts": requirements,
        }
        # Encode as base64 PAYMENT-REQUIRED header (x402 v2 protocol)
        encoded = base64.b64encode(json.dumps(payment_required).encode()).decode()
        return JSONResponse(
            status_code=402,
            content={
                **payment_required,
                "error": "X-PAYMENT header is required",
            },
            headers={
                "WWW-Authenticate": "X-PAYMENT",
                "PAYMENT-REQUIRED": encoded,
            },
        )

    @staticmethod
    async def verify_payment(payment_header: str, requirements: dict) -> bool:
        """Verify x402 payment via thirdweb (no settlement)."""
        try:
            # x402 v1 sends raw JSON in X-PAYMENT, v2 sends base64-encoded JSON in PAYMENT-SIGNATURE
            try:
                payment_payload = json.loads(payment_header)
            except json.JSONDecodeError:
                try:
                    payment_payload = json.loads(base64.b64decode(payment_header))
                except Exception:
                    logger.error("Invalid x402 payment header: not valid JSON or base64")
                    return False

            payload = {
                "x402Version": 2,
                "paymentPayload": payment_payload,
                "paymentRequirements": requirements,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{THIRDWEB_X402_BASE}/verify",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-secret-key": config.THIRDWEB_SECRET_KEY,
                    },
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        is_valid = data.get("isValid", False)
                        if not is_valid:
                            logger.warning(f"thirdweb verify returned invalid: {json.dumps(data)}")
                        return is_valid
                    else:
                        error_text = await response.text()
                        logger.error(f"thirdweb verify error: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"x402 payment verification failed: {e}", exc_info=True)
            return False

    @staticmethod
    async def settle_payment(payment_header: str, requirements: dict, actual_amount: float) -> bool:
        """Settle x402 payment via thirdweb (actually collect the funds)."""
        try:
            try:
                payment_payload = json.loads(payment_header)
            except json.JSONDecodeError:
                try:
                    payment_payload = json.loads(base64.b64decode(payment_header))
                except Exception:
                    logger.error("Invalid x402 payment header for settlement")
                    return False

            # Shallow copy to avoid mutating caller's dict
            actual_amount_micro = str(int(actual_amount * 1_000_000))
            settle_requirements = {**requirements}
            # Only override maxAmountRequired for upto scheme; exact scheme already has the fixed amount
            if requirements.get("scheme", "upto") != "exact":
                settle_requirements["maxAmountRequired"] = actual_amount_micro

            headers = {
                "Content-Type": "application/json",
                "x-secret-key": config.THIRDWEB_SECRET_KEY,
            }
            if config.THIRDWEB_VAULT_ACCESS_TOKEN:
                headers["x-vault-access-token"] = config.THIRDWEB_VAULT_ACCESS_TOKEN

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{THIRDWEB_X402_BASE}/settle",
                    json={
                        "x402Version": 2,
                        "paymentPayload": payment_payload,
                        "paymentRequirements": settle_requirements,
                        "waitUntil": "confirmed",
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 200:
                        logger.info(f"x402 payment settled ({actual_amount_micro} micro-USDC)")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"thirdweb settle error: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"x402 payment settlement failed: {e}", exc_info=True)
            return False

    async def refresh_prices(self):
        """Pull per-token prices from inference backend."""
        try:
            async with aiohttp.ClientSession() as session:
                session.headers["x-admin-token"] = config.BACKEND_SECRET_TOKEN
                async with session.get(f"{config.BACKEND_API_URL}/x402/prices") as response:
                    if response.status == 200:
                        self.prices = await response.json()
                        logger.debug(f"Refreshed x402 prices: {len(self.prices)} models")
                    else:
                        logger.error(f"Error fetching x402 prices: {response.status}")
        except Exception as e:
            logger.error(f"Exception fetching x402 prices: {e}", exc_info=True)


x402_manager = X402Manager()
