import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.account_manager import AccountManager
from src.auth import router as auth_router
from src.tasks import sync_metrics

account_manager = AccountManager()


async def run_jobs():
    while True:
        await asyncio.sleep(10)
        await sync_metrics()
        await account_manager.load_active_accounts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting node...")
    await account_manager.load_active_accounts()
    asyncio.create_task(run_jobs())
    yield


app = FastAPI(title="LibertAI node service", lifespan=lifespan)
app.include_router(auth_router)
