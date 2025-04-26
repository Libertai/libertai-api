import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.api_keys import KeysManager
from src.auth import router as auth_router
from src.health import server_health_monitor
from src.model import router as model_router
from src.proxy import router as proxy_router

keys_manager = KeysManager()


async def run_jobs():
    while True:
        await keys_manager.refresh_keys()
        await server_health_monitor.check_all_servers()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server...")
    asyncio.create_task(run_jobs())
    yield


app = FastAPI(title="LibertAI backend service", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(model_router)
app.include_router(proxy_router)
