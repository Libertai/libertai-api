from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config

router = APIRouter(tags=["Auth service"])


@router.get("/libertai/models")
async def models_list():
    # TODO: filter out instances that are down
    models = config.MODELS
    data = {}
    for model_name in models:
        servers = []
        for item in models[model_name]:
            servers.append(item.url)

        data[model_name] = {"servers": servers}

    return JSONResponse(content=data)
