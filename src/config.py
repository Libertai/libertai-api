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


class _Config:
    BACKEND_API_URL: str
    BACKEND_SECRET_TOKEN: str
    MODELS: dict[str, list[str]]
    SERVER_CAPACITY: dict[str, int]
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
    ALEPH_AGGREGATE_ADDRESS: str
    REDIS_URL: str
    SEARCH_SERVICE_URL: str
    FREE_SOFT_LOAD: int
    FREE_HARD_LOAD: int
    MAX_BODY_SIZE_MB: int
    METRICS_TOKEN: str
    PROMETHEUS_MULTIPROC_DIR: str

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
        # The Aleph aggregate holding LTAI_PRICING (models metadata + redirections),
        # read by src/aleph.py and published by scripts/set_model_redirect.py.
        # Blank or whitespace-only is treated like unset so an empty .env value
        # can't break the fetch.
        self.ALEPH_AGGREGATE_ADDRESS = (
            os.getenv("ALEPH_AGGREGATE_ADDRESS", "").strip() or "0x238224C744F4b90b4494516e074D2676ECfC6803"
        )
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

        # Token required to scrape /metrics (?token=...). Metrics expose business
        # behavior, so the endpoint stays locked (401) until this is configured.
        self.METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")

        # Optional prometheus_client multiprocess mode (WEB_CONCURRENCY > 1): a
        # writable directory where each worker writes its counter/histogram
        # files and every scrape merges them. prometheus_client enables mmap
        # mode on the variable's *presence* (even empty), so blank values —
        # e.g. a shipped "PROMETHEUS_MULTIPROC_DIR=" line — are normalized to
        # unset here, before prometheus_client is imported; otherwise workers
        # would write stray db files to the cwd while the scrape renders the
        # single-process registry. The app never clears the directory:
        # parallel workers would wipe each other's files, so the deployment
        # (e.g. tmpfs) must start empty, and the app does not create it — a
        # typo fails loudly at startup instead of silently writing elsewhere.
        multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "").strip()
        if multiproc_dir:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
        else:
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        # prometheus_client also presence-checks the deprecated lowercase
        # spelling; a stale value there would flip workers to mmap mode while
        # this config (and the render branch) stay unset, so the merge would
        # silently never happen. Drop it so both mode checks agree.
        os.environ.pop("prometheus_multiproc_dir", None)
        self.PROMETHEUS_MULTIPROC_DIR = multiproc_dir

        # Load models configuration from environment variable or file
        models_config = os.getenv("MODELS_CONFIG")
        self.MODELS = {}
        # Optional per-server overflow capacity: a models.json entry may be
        # {"url": "...", "capacity": N} instead of a plain URL string. Capacity
        # is how many concurrent requests a server absorbs before the gateway
        # overflows to the next configured server (proxy.py saturation sort).
        # Default 1 keeps the pre-capacity behavior for unconfigured servers.
        self.SERVER_CAPACITY = {}

        # Configure logging
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.LOG_LEVEL = getattr(logging, log_level_str, logging.INFO)

        if models_config:
            try:
                with open(models_config) as f:
                    models_data = json.load(f)
                    for model_name, servers in models_data.items():
                        parsed_servers: list[str] = []
                        for server in servers:
                            if isinstance(server, dict):
                                url = server.get("url")
                                if not isinstance(url, str) or not url.strip():
                                    logging.getLogger(__name__).error(
                                        f"Invalid server entry {server!r} for model "
                                        f"'{model_name}' in {models_config}: missing url; skipping"
                                    )
                                    continue
                                parsed_servers.append(url)
                                capacity = server.get("capacity", 1)
                                if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                                    logging.getLogger(__name__).warning(
                                        f"Invalid capacity {capacity!r} for {url} in {models_config}; "
                                        f"using 1"
                                    )
                                else:
                                    self.SERVER_CAPACITY[url] = capacity
                            else:
                                parsed_servers.append(server)
                        self.MODELS[model_name.lower()] = parsed_servers
            except json.JSONDecodeError as error:
                logging.getLogger(__name__).error(f"Error parsing {models_config}: {error}", exc_info=True)


config = _Config()
