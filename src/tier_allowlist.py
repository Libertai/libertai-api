from fastapi.responses import JSONResponse

from src.api_keys import KeysManager
from src.config import config
from src.errors import model_not_in_plan_response
from src.logger import setup_logger

keys_manager = KeysManager()
logger = setup_logger(__name__)


def _matches(model: str, patterns: list[str]) -> bool:
    return any(model.startswith(p[:-1]) if p.endswith("*") else model == p for p in patterns)


def model_not_in_plan(api_key: str | None, requested: str, resolved: str) -> JSONResponse | None:
    """403 response when the key's tier has an allowlist that excludes the model, else None.

    `resolved` is the model actually served (redirects applied, `-thinking`
    reduced to its base), so a thinking variant of a blocked model stays
    blocked and an alias redirecting to an allowed model passes. `requested`
    is only matched exactly, so a listed alias passes whatever it redirects to,
    but a glob never reaches through a redirect to an unlisted model.

    Unknown keys and tiers without an entry fail open, like the free-tier gate
    in src/proxy.py: sync skew must not lock paying users out.
    """
    if api_key is None or not keys_manager.key_exists(api_key):
        return None
    tier = (keys_manager.tier(api_key) or "").lower()
    allowed = config.TIER_MODEL_ALLOWLIST.get(tier)
    if allowed is None:
        return None
    if _matches(resolved.lower(), allowed) or requested.lower() in allowed:
        return None
    logger.info(f"Tier '{tier}' request to model '{requested}' (resolved '{resolved}') blocked: not in plan")
    return model_not_in_plan_response()
