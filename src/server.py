import asyncio
from contextlib import asynccontextmanager

# Use uvloop for better async performance
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # Fall back to default asyncio event loop

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from src.aleph import aleph_service
from src.aleph import close_http_client as close_aleph_http_client
from src.aleph_credits import router as aleph_credits_router
from src.api_keys import KeysManager
from src.api_keys import close_http_client as close_keys_http_client
from src.auth import router as auth_router
from src.config import config
from src.constants import JOB_INTERVAL_SECONDS
from src.errors import body_too_large_response
from src.health import close_http_client as close_health_http_client
from src.health import server_health_monitor
from src.leader import leader
from src.logger import setup_logger
from src.model import router as model_router
from src.proxy import close_http_client
from src.proxy import router as proxy_router
from src.redis_client import close_redis
from src.search import close_http_client as close_search_http_client
from src.search import router as search_router
from src.x402 import close_http_client as close_x402_http_client
from src.x402 import x402_manager

# The Telegram bot now runs as its own dokploy service (replicas: 1, entrypoint
# `python -m src.bot`), so the web replicas no longer poll Telegram or run the
# alert/supervisor loops. They still need leader election for run_jobs.

keys_manager = KeysManager()
logger = setup_logger(__name__)


class _BodySizeLimitMiddleware:
    """Reject oversized request bodies before the body is ever read.

    The proxy buffers the entire request body in memory, so an uncapped upload
    is a memory-exhaustion vector on a public API. Reads only the
    content-length header from the scope, so it stays cheap on the hot path.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and config.MAX_BODY_SIZE_MB > 0:
            content_length = Headers(scope=scope).get("content-length")
            max_bytes = config.MAX_BODY_SIZE_MB * 1024 * 1024
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                response = body_too_large_response(config.MAX_BODY_SIZE_MB)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Set to True after first successful job cycle
_ready = False


def _has_authoritative_state() -> bool:
    """Keys and model metadata must be loaded before a replica reports ready."""
    return bool(keys_manager.keys and aleph_service.models_loaded)


async def _refresh_state():
    """Leader refreshes upstream state; every replica syncs from Redis."""
    if leader.is_leader:
        await keys_manager.refresh_keys()
        # Refresh model metadata before the slow per-server health sweep so
        # /v1/models and /openrouter/models are enriched from the first cycle.
        await aleph_service.refresh()
        # Aleph unreachable? Serve the last snapshot published to Redis instead of
        # stalling readiness while followers recover from the same snapshot.
        if not aleph_service.models_loaded:
            await aleph_service.sync_from_redis()
        await server_health_monitor.check_all_servers()
        await x402_manager.refresh_prices()
    else:
        await keys_manager.sync_from_redis()
        await aleph_service.sync_from_redis()
        await server_health_monitor.sync_from_redis()
        await x402_manager.sync_from_redis()


async def run_jobs():
    """Periodic jobs. Leader refreshes upstream state; every replica syncs from Redis."""
    global _ready
    while True:
        try:
            await _refresh_state()
            # Only mark ready once we actually have authoritative data; otherwise
            # replicas would serve 401s against an empty key set or empty model
            # listings against a missing Aleph snapshot during cold start.
            if _has_authoritative_state():
                _ready = True
        except Exception as e:
            logger.error(f"Error in run_jobs: {e}", exc_info=True)
        await asyncio.sleep(JOB_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    leader_task = asyncio.create_task(leader.run())
    jobs_task = asyncio.create_task(run_jobs())

    try:
        yield
    finally:
        await leader.shutdown()
        for t in (leader_task, jobs_task):
            t.cancel()
        await asyncio.gather(leader_task, jobs_task, return_exceptions=True)
        await close_http_client()
        await close_search_http_client()
        await close_health_http_client()
        await close_keys_http_client()
        await close_x402_http_client()
        await close_aleph_http_client()
        await close_redis()


app = FastAPI(title="LibertAI API", lifespan=lifespan)


app.add_middleware(_BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check that reports ready only after first full initialization cycle."""
    if not _ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "keys_loaded": len(keys_manager.keys) > 0,
                "models_loaded": aleph_service.models_loaded,
                "prices_loaded": len(x402_manager.prices) > 0,
            },
        )

    healthy_models = {model: urls for model, urls in server_health_monitor.healthy_model_urls.items() if urls}

    return {
        "status": "ok",
        "keys_loaded": len(keys_manager.keys) > 0,
        "models_loaded": aleph_service.models_loaded,
        "healthy_models": len(healthy_models),
        "prices_loaded": len(x402_manager.prices) > 0,
    }


app.include_router(auth_router)
app.include_router(model_router)
app.include_router(aleph_credits_router)
app.include_router(search_router)
app.include_router(proxy_router)
