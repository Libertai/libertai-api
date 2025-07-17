import random
from http import HTTPStatus
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import ServerConfig, config
from src.health import server_health_monitor
from src.logger import setup_logger

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(timeout=600.0)  # 10 minutes

logger = setup_logger(__name__)


class ProxyRequest(BaseModel):
    model: str

    class Config:
        extra = "allow"  # Allow extra fields


def select_server(model_name: str) -> Optional[ServerConfig]:
    """
    Select a server based on weights.

    Args:
        model_name: Name of the model to use

    Returns:
        Selected server or None if no servers available
    """

    model = model_name.lower()
    if model not in config.MODELS or not config.MODELS[model]:
        return None

    servers = config.MODELS[model]
    healthy_model_urls = server_health_monitor.get_healthy_model_urls()

    # Check if there are any healthy servers for this model
    if model not in healthy_model_urls or not healthy_model_urls[model]:
        return None

    # Filter out unhealthy servers
    servers = [server for server in servers if server.url in healthy_model_urls[model]]

    if not servers:
        return None

    # Calculate total weight
    total_weight = sum(server.weight for server in servers)

    if total_weight == 0:
        return random.choice(servers)

    # Select based on weight
    r = random.uniform(0, total_weight)
    current_weight = 0

    for server in servers:
        current_weight += server.weight
        if r <= current_weight:
            return server

    return servers[-1]  # Fallback to last server


@router.post("/{full_path:path}")
async def proxy_request(
    full_path: str,
    request: Request,
    proxy_request_data: ProxyRequest,
):
    # Get model from request
    model_name = proxy_request_data.model

    logger.debug(f"Received proxy request to {full_path} for model {model_name}")

    # Select server
    server = select_server(model_name)
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
            url = f"{server.url}/{full_path}"

            response: httpx.Response = await client.post(
                url, content=body, headers=headers, params=request.query_params
            )
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error forwarding request: {str(e)}"
            )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )
