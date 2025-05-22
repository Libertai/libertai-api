import random
from http import HTTPStatus
from typing import Optional, Union

import aiohttp
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import ServerConfig, config
from src.health import server_health_monitor

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()


class ProxyRequest(BaseModel):
    model: str
    prefer_gpu: bool = False

    class Config:
        extra = "allow"  # Allow extra fields


def select_server(model_name: str, prefer_gpu: bool = False) -> Optional[ServerConfig]:
    """
    Select a server based on weights and GPU preference.

    Args:
        model_name: Name of the model to use
        prefer_gpu: Whether to prefer GPU servers

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

    # Filter by GPU if preferred
    if prefer_gpu:
        gpu_servers = [s for s in servers if s.gpu]
        if gpu_servers:
            servers = gpu_servers

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


async def process_response(
    response: aiohttp.ClientResponse,
) -> Union[Response, StreamingResponse]:
    """
    Process the response from the upstream server.

    Args:
        response: The response from the upstream server

    Returns:
        Either a Response or StreamingResponse object with the processed data
    """
    content_type = response.headers.get("Content-Type", "")

    # Handle JSON responses
    if "application/json" in content_type:
        try:
            # Return processed JSON response
            return Response(
                content=await response.read(),
                status_code=response.status,
                headers=response.headers,
                media_type=content_type,
            )
        except Exception:
            # If something fails, fall back to streaming
            pass

    # For non-JSON responses, use streaming
    return StreamingResponse(
        content=response.content.iter_any(),
        status_code=response.status,
        headers=dict(response.headers),
    )


@router.post("/{full_path:path}")
async def proxy_request(
    full_path: str,
    request: Request,
    proxy_request_data: ProxyRequest,
):
    # Get model from request
    model_name = proxy_request_data.model

    # Select server
    server = select_server(model_name, proxy_request_data.prefer_gpu)
    if not server:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No server available for model {model_name}",
        )

    # Get the original request body
    body = await request.json()

    # Forward the request to the selected server
    async with aiohttp.ClientSession() as session:
        try:
            forwarded_headers = {}
            headers = dict(request.headers)
            forwarded_headers["authorization"] = headers["authorization"]
            # Forward the request to the selected server
            url = f"{server.url}/{full_path}"

            async with session.request(
                method=request.method,
                url=url,
                json=body,
                headers=forwarded_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                # Process and return the response
                return await process_response(response)
        except Exception as e:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Error forwarding request: {str(e)}"
            )
