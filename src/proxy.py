import json
from http import HTTPStatus

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Cookie
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import config
from src.health import server_health_monitor
from src.logger import setup_logger

router = APIRouter(tags=["Proxy"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(timeout=600.0)  # 10 minutes

logger = setup_logger(__name__)


class ProxyRequest(BaseModel):
    model: str

    class Config:
        extra = "allow"  # Allow extra fields


@router.post("/{full_path:path}")
async def proxy_request(
    full_path: str,
    request: Request,
    proxy_request_data: ProxyRequest,
    preferred_instances: str = Cookie(default="{}"),  # JSON-encoded map
):
    # Get model from request
    model_name = proxy_request_data.model

    logger.debug(f"Received proxy request to {full_path} for model {model_name}")

    try:
        preferred_instances_map = json.loads(preferred_instances)
    except json.JSONDecodeError:
        preferred_instances_map = {}

    preferred_server: str | None = preferred_instances_map.get(model_name)

    model = model_name.lower()
    if model not in config.MODELS or not config.MODELS[model]:
        return None

    # Select server from the health and metrics monitoring
    server = server_health_monitor.get_least_busy_server(model, preferred_server)

    if not server:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No server available for model {model_name}",
        )

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Clean up headers
    headers.pop("host", None)

    # Forward the request
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Forward the request to the selected server
            url = f"{server}/{full_path}"

            response: httpx.Response = await client.post(
                url, content=body, headers=headers, params=request.query_params
            )
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error forwarding request: {str(e)}"
            )

    proxy_response = Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )

    # Update or set the preferred instance for this model
    preferred_instances_map[model_name] = server
    updated_cookie_value = json.dumps(preferred_instances_map)

    proxy_response.set_cookie(
        key="preferred_instances", value=updated_cookie_value, max_age=1800, httponly=True, secure=True, samesite="lax"
    )

    return proxy_response
