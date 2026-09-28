import json
import logging
import os

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    """Read an integer env var; treat missing/blank/malformed values as the default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).error(f"Invalid {name}='{raw}': falling back to {default}")
        return default


def _tier_model_allowlist_env(name: str) -> dict[str, list[str]]:
    """Read a JSON {tier: [model, ...]} env var, lowercased to match the case-normalised
    tier and model lookups in src/tier_allowlist.py. Missing or malformed values
    disable the allowlist."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not all(
            isinstance(models, list) and all(isinstance(m, str) for m in models) for models in data.values()
        ):
            raise ValueError("expected a JSON object of tier -> list of model names")
    except ValueError as error:
        # json.JSONDecodeError is a ValueError too
        logging.getLogger(__name__).error(f"Invalid {name}: {error}; the tier model allowlist is disabled")
        return {}
    return {str(tier).lower(): [m.lower() for m in models] for tier, models in data.items()}


class _Config:
    BACKEND_API_URL: str
    BACKEND_SECRET_TOKEN: str
    MODELS: dict[str, list[str]]
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    TELEGRAM_TOPIC_ID: str
    PRIVATE_KEY: str
    X402_API_KEY: str
    X402_WALLET_ADDRESS: str
    X402_SERVER_WALLET_ADDRESS: str
    THIRDWEB_SECRET_KEY: str
    PUBLIC_BASE_URL: str
    THIRDWEB_VAULT_ACCESS_TOKEN: str
    ALEPH_SENDER_PRIVATE_KEY: str
    REDIS_URL: str
    SEARCH_SERVICE_URL: str
    FREE_SOFT_LOAD: int
    FREE_HARD_LOAD: int
    MAX_BODY_SIZE_MB: int
    TIER_MODEL_ALLOWLIST: dict[str, list[str]]

    LOG_LEVEL: int

    def __init__(self):
        load_dotenv()

        self.BACKEND_API_URL = os.getenv("BACKEND_API_URL")
        self.BACKEND_SECRET_TOKEN = os.getenv("BACKEND_SECRET_TOKEN")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
        self.TELEGRAM_TOPIC_ID = os.getenv("TELEGRAM_TOPIC_ID", "")
        self.PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
        self.X402_API_KEY = os.getenv("X402_API_KEY", "")
        self.X402_WALLET_ADDRESS = os.getenv("X402_WALLET_ADDRESS", "")
        self.X402_SERVER_WALLET_ADDRESS = os.getenv("X402_SERVER_WALLET_ADDRESS", "")
        self.THIRDWEB_SECRET_KEY = os.getenv("THIRDWEB_SECRET_KEY", "")
        self.PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        self.THIRDWEB_VAULT_ACCESS_TOKEN = os.getenv("THIRDWEB_VAULT_ACCESS_TOKEN", "")
        self.ALEPH_SENDER_PRIVATE_KEY = os.getenv("ALEPH_SENDER_PRIVATE_KEY", "")
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "https://search.libertai.io").rstrip("/")

        # Free-tier admission gate thresholds (src/proxy.py): aggregate inflight
        # requests across the requested model's servers. At the soft threshold
        # free-tier keys wait (bounded) for the pool to drain; at the hard
        # threshold they are rejected outright. Paid tiers bypass the gate.
        # Defaults are sized for a pool of ~2 boxes at ~25 concurrent
        # generations each, meant to be tuned via env against real load data;
        # raise both thresholds to disable the gate (a high hard threshold alone
        # only removes the rejection — the soft wait still applies).
        self.FREE_SOFT_LOAD = _int_env("FREE_SOFT_LOAD", 25)
        self.FREE_HARD_LOAD = _int_env("FREE_HARD_LOAD", 50)
        # Fails safe either way, but an operator who fat-fingers the values should
        # hear about it: SOFT > HARD disables the wait phase, HARD <= 0 rejects
        # every free request, and SOFT <= 0 forces the wait even on an idle pool.
        if not (0 < self.FREE_SOFT_LOAD <= self.FREE_HARD_LOAD):
            logging.getLogger(__name__).warning(
                f"FREE_SOFT_LOAD ({self.FREE_SOFT_LOAD}) / FREE_HARD_LOAD ({self.FREE_HARD_LOAD}): "
                f"expected 0 < SOFT <= HARD; the gate is misconfigured"
            )
        # Request body size cap (src/server.py middleware): the proxy buffers the
        # entire body in memory, so an uncapped upload is a memory-exhaustion
        # vector. 0 or negative disables the cap.
        self.MAX_BODY_SIZE_MB = _int_env("MAX_BODY_SIZE_MB", 100)
        # Per-tier model allowlist (src/tier_allowlist.py), as JSON, e.g.
        # TIER_MODEL_ALLOWLIST='{"liberclaw:free": ["qwen3.6-35b-a3b", "search/*"]}'.
        # A key whose tier has an entry may only use the listed models; a trailing
        # "*" matches by prefix. The search endpoints count as the models
        # "search/search" and "search/fetch", so a restricted tier needs "search/*"
        # to keep web search. Tiers without an entry are unrestricted, and the
        # default (unset) turns the check off.
        self.TIER_MODEL_ALLOWLIST = _tier_model_allowlist_env("TIER_MODEL_ALLOWLIST")

        # Load models configuration from environment variable or file
        models_config = os.getenv("MODELS_CONFIG")
        self.MODELS = {}

        # Configure logging
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.LOG_LEVEL = getattr(logging, log_level_str, logging.INFO)

        if models_config:
            try:
                with open(models_config) as f:
                    models_data = json.load(f)
                    for model_name, servers in models_data.items():
                        self.MODELS[model_name.lower()] = servers
            except json.JSONDecodeError as error:
                logging.getLogger(__name__).error(f"Error parsing {models_config}: {error}", exc_info=True)


config = _Config()
