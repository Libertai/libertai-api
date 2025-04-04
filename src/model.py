from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config

router = APIRouter(tags=["Auth service"])


@router.get("/model/list")
async def models_list():
    models = config.MODELS
    data = {
        "models": [],
        "details": {}
    }
    for model_name in models:
        model = models[model_name][0]
        print(model)
        data["models"].append(model_name)
        data["details"][model_name] = {
            "url": model.url,
            "type": model.type,
            "api_type": model.api_type,
            "completion_path": model.completion_path,
            "prompt_format": model.prompt_format,
        }
    return JSONResponse(content=data)
