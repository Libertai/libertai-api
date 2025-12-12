import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config

router = APIRouter(tags=["Models"])


@router.get("/libertai/models")
async def models_list():
    # Get all configured servers
    data = {}
    for model_name, servers in config.MODELS.items():
        data[model_name] = {"servers": servers}

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
