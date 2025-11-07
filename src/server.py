import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import KeysManager
from src.auth import router as auth_router
from src.health import server_health_monitor
from src.logger import setup_logger
from src.model import router as model_router
from src.proxy import router as proxy_router
from src.telegram import telegram_reporter

keys_manager = KeysManager()
logger = setup_logger(__name__)

# Constants
HEALTH_CHECK_INTERVAL = 30  # seconds
TELEGRAM_REPORT_INTERVAL = 1800  # 30 minutes


async def run_jobs():
    """Run periodic jobs for key refresh and health checks."""
    while True:
        await keys_manager.refresh_keys()
        await server_health_monitor.check_all_servers()
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def run_telegram_reporting():
    """Run hourly Telegram health reporting."""
    while True:
        try:
            await telegram_reporter.send_health_report()
        except Exception as e:
            logger.error(f"Error in Telegram reporting: {e}")

        # Sleep until the next hour
        await asyncio.sleep(TELEGRAM_REPORT_INTERVAL)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Start background tasks
    asyncio.create_task(run_jobs())
    asyncio.create_task(run_telegram_reporting())
    yield


app = FastAPI(title="LibertAI API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(model_router)
app.include_router(proxy_router)
