from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config
from src.health import server_health_monitor

router = APIRouter(tags=["Auth service"])


@router.get("/libertai/models")
async def models_list():
    # Get only healthy servers from the health monitor
    healthy_servers = server_health_monitor.get_healthy_model_urls()

    data = {}
    for model_name in config.MODELS:
        data[model_name] = {"servers": healthy_servers[model_name]}

    return JSONResponse(content=data)
