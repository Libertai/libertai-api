import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.account import router as account_router
from src.account_manager import AccountManager
from src.auth import router as auth_router
from src.token import router as token_router
from src.utils.metrics import sync_metrics

account_manager = AccountManager()


async def run_jobs():
    while True:
        await asyncio.sleep(300)
        await sync_metrics()
        await account_manager.load_active_accounts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server...")
    await account_manager.load_active_accounts()
    asyncio.create_task(run_jobs())
    yield

app = FastAPI(title="LibertAI backend service", lifespan=lifespan)

origins = [
    "https://chat.libertai.io",
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
#app.include_router(token_router)
#app.include_router(account_router)
