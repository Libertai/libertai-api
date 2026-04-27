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
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import KeysManager
from src.auth import router as auth_router
from src.health import server_health_monitor
from src.leader import leader
from src.logger import setup_logger
from src.model import router as model_router
from src.aleph_credits import router as aleph_credits_router
from src.proxy import router as proxy_router, close_http_client
from src.redis_client import close_redis
from src.search import router as search_router, close_http_client as close_search_http_client
from src.telegram import telegram_reporter
from src.aleph import aleph_service
from src.x402 import x402_manager

keys_manager = KeysManager()
logger = setup_logger(__name__)

# Constants
HEALTH_CHECK_INTERVAL = 30  # seconds
TELEGRAM_REPORT_INTERVAL = 1800  # 30 minutes

# Set to True after first successful job cycle
_ready = False


async def run_jobs():
    """Periodic jobs. Leader refreshes upstream state; every replica syncs from Redis."""
    global _ready
    while True:
        try:
            if leader.is_leader:
                await keys_manager.refresh_keys()
                await server_health_monitor.check_all_servers()
                await x402_manager.refresh_prices()
                await aleph_service.refresh()
            else:
                await keys_manager.sync_from_redis()
                await server_health_monitor.sync_from_redis()
                await x402_manager.sync_from_redis()
                await aleph_service.sync_from_redis()
            # Only mark ready once we actually have authoritative data; otherwise
            # followers would serve 401s against an empty key set during cold start.
            if keys_manager.keys:
                _ready = True
        except Exception as e:
            logger.error(f"Error in run_jobs: {e}", exc_info=True)
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def run_telegram_reporting():
    """Leader-only: periodic Telegram health reporting."""
    while True:
        if leader.is_leader:
            try:
                await telegram_reporter.send_health_report()
            except Exception as e:
                logger.error(f"Error in Telegram reporting: {e}")
        await asyncio.sleep(TELEGRAM_REPORT_INTERVAL)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    leader.on_acquire(telegram_reporter.start_bot)
    leader.on_release(telegram_reporter.stop_bot)

    leader_task = asyncio.create_task(leader.run())
    jobs_task = asyncio.create_task(run_jobs())
    tg_task = asyncio.create_task(run_telegram_reporting())

    try:
        yield
    finally:
        # leader.shutdown() fires on_release → stop_bot(), so no explicit stop_bot needed.
        await leader.shutdown()
        for t in (leader_task, jobs_task, tg_task):
            t.cancel()
        await asyncio.gather(leader_task, jobs_task, tg_task, return_exceptions=True)
        await close_http_client()
        await close_search_http_client()
        await close_redis()


app = FastAPI(title="LibertAI API", lifespan=lifespan)


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
        return JSONResponse(status_code=503, content={"status": "starting"})

    healthy_models = {model: urls for model, urls in server_health_monitor.healthy_model_urls.items() if urls}

    return {
        "status": "ok",
        "keys_loaded": len(keys_manager.keys) > 0,
        "healthy_models": len(healthy_models),
        "prices_loaded": len(x402_manager.prices) > 0,
    }


app.include_router(auth_router)
app.include_router(model_router)
app.include_router(aleph_credits_router)
app.include_router(search_router)
app.include_router(proxy_router)
