from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import config

router = APIRouter(tags=["Auth service"])


@router.get("/libertai/models")
async def models_list():
    models = config.MODELS
    data = {
    }
    for model_name in models:
        #model = models[model_name][0]
        servers = []
        for item in models[model_name]:
            servers .append(item.url)

        data[model_name] = {
            "servers": servers
        }

        """
        data["details"][model_name] = {
            "url": "https://api.libertai.io",
            "type": model.type,
            "api_type": model.api_type,
            "completion_path": model.completion_paths,
            "prompt_format": model.prompt_format,
            "servers": servers
        }
        """
        
    return JSONResponse(content=data)
