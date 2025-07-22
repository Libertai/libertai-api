import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config
from src.health import server_health_monitor

router = APIRouter(tags=["Models"])


@router.get("/libertai/models")
async def models_list():
    # Get only healthy servers from the health monitor
    healthy_servers = server_health_monitor.get_healthy_model_urls()

    data = {}
    for model_name in config.MODELS:
        data[model_name] = {"servers": healthy_servers[model_name]}

    return JSONResponse(content=data)


@router.get("/v1/models")
async def openai_models_list():
    """
    Returns a list of available models in OpenAI API format.
    """
    current_timestamp = int(time.time())

    models_data = []
    for model_name in config.MODELS.keys():
        model_entry = {"id": model_name, "object": "model", "created": current_timestamp, "owned_by": "libertai"}
        models_data.append(model_entry)

    response = {"object": "list", "data": models_data}

    return JSONResponse(content=response)
