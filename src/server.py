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
from src.errors import body_too_large_response, client_disconnected_response
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
    """Reject oversized request bodies before the app buffers them.

    The proxy buffers the entire request body in memory, so an uncapped upload
    is a memory-exhaustion vector on a public API. A parseable content-length
    is checked from the scope (cheap on the hot path); bodies without one
    (chunked / HTTP-2 framing) are drained with the same cap and replayed to
    the app, so neither path can be buffered unboundedly.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and config.MAX_BODY_SIZE_MB > 0:
            max_bytes = config.MAX_BODY_SIZE_MB * 1024 * 1024
            content_length = _content_length_int(Headers(scope=scope).get("content-length") or "")
            if content_length is not None and content_length > max_bytes:
                await self._reject(scope, receive, send, True)
                return
            if content_length is None:
                chunks, over_cap = await self._drain(receive, max_bytes)
                if chunks is None:
                    await self._reject(scope, receive, send, over_cap)
                    return
                receive = _replay_receive(chunks, receive)
        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send, over_cap: bool) -> None:
        if over_cap:
            response = body_too_large_response(config.MAX_BODY_SIZE_MB)
        else:
            response = client_disconnected_response()
        await response(scope, receive, send)

    async def _drain(self, receive: Receive, max_bytes: int) -> tuple[list[bytes] | None, bool]:
        """Read the body up to the cap.

        Returns (chunks, over_cap): over_cap True when the cap is exceeded;
        (None, False) on a mid-drain client disconnect — forwarding a truncated
        body would be worse.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None, False
            body = message.get("body", b"")
            total += len(body)
            if total > max_bytes:
                return None, True
            chunks.append(body)
            if not message.get("more_body"):
                return chunks, False


def _content_length_int(raw: str) -> int | None:
    """ASCII-digits-only parse: isdigit() also accepts Unicode digits that int() rejects."""
    if not raw.isascii() or not raw.isdigit():
        return None
    return int(raw)


def _replay_receive(chunks: list[bytes], real_receive: Receive) -> Receive:
    """Return a receive that yields the drained chunks back so the app reads
    the body it would have read without the middleware.

    After exhaustion it delegates to the original receive: StreamingResponse
    listens for disconnect concurrently with the stream, so a synthetic
    disconnect here would cancel the response right after the headers.
    """
    state = {"index": 0}

    async def receive():
        if state["index"] < len(chunks):
            chunk = chunks[state["index"]]
            state["index"] += 1
            return {"type": "http.request", "body": chunk, "more_body": state["index"] < len(chunks)}
        return await real_receive()

    return receive


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
