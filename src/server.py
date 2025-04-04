import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import KeysManager
from src.auth import router as auth_router
from src.model import router as model_router
from src.proxy import router as proxy_router

keys_manager = KeysManager()


async def run_jobs():
    while True:
        await asyncio.sleep(300)
        await keys_manager.refresh_keys()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server...")
    await keys_manager.refresh_keys()
    asyncio.create_task(run_jobs())
    yield

app = FastAPI(title="LibertAI backend service", lifespan=lifespan)

origins = [
    "https://chat.libertai.io",
    "http://localhost:9000",
    "https://bafybeid7ag5d4it32ylu5nkqw6lzsddtqzhhs3egu2pskwcadbz6jzmxby.ipfs.aleph.sh"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(model_router)
app.include_router(proxy_router)
