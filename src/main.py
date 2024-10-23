import asyncio

from fastapi import FastAPI

from src.auth import router as auth_router
from src.tasks import sync_metrics
from src.token import router as token_router
from src.token_manager import TokenManager

app = FastAPI(title="LibertAI backend service")
app.include_router(auth_router)
app.include_router(token_router)

token_manager = TokenManager()


async def run_jobs():
    while True:
        await asyncio.sleep(60)
        await token_manager.load_active_tokens()
        await sync_metrics()


def main():
    asyncio.create_task(token_manager.load_active_tokens())
    asyncio.create_task(run_jobs())


if __name__.split(".")[-1] == "main":
    main()
