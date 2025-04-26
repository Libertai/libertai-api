import random
from http import HTTPStatus
from typing import Annotated, Any, Dict, Optional, Tuple, Union

import aiohttp
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import ServerConfig, config
from src.health import server_health_monitor
from src.tasks import report_usage_event_task

router = APIRouter(tags=["Proxy service"])
keys_manager = KeysManager()
security = HTTPBearer()


class ProxyRequest(BaseModel):
    model: str
    prefer_gpu: bool = False

    class Config:
        extra = "allow"  # Allow extra fields


class Usage(BaseModel):
    key: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    model_name: str


class UserContext(BaseModel):
    key: str
    model_name: str


def select_server(user_context: UserContext, prefer_gpu: bool = False) -> Optional[ServerConfig]:
    """
    Select a server based on weights and GPU preference.

    Args:
        user_context: Context
        prefer_gpu: Whether to prefer GPU servers

    Returns:
        Selected server or None if no servers available
    """

    model = user_context.model_name.lower()
    if model not in config.MODELS or not config.MODELS[model]:
        return None

    servers = config.MODELS[model]
    healthy_server_urls = server_health_monitor.get_healthy_servers()
    # Filter out unhealthy servers
    servers = [server for server in servers if server.url in healthy_server_urls]

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
    response: aiohttp.ClientResponse, user_context: UserContext, background_tasks: BackgroundTasks | None = None
) -> Union[Response, StreamingResponse]:
    """
    Process the response from the upstream server and extract token information.

    Args:
        response: The response from the upstream server
        user_context: Context
        background_tasks: Tasks

    Returns:
        Either a Response or StreamingResponse object with the processed data
    """
    content_type = response.headers.get("Content-Type", "")

    # Handle JSON responses - extract token information
    if "application/json" in content_type:
        try:
            # Get response JSON to extract token counts
            response_json = await response.json()

            if config.REPORT_USAGE:
                # Extract usage information
                try:
                    if background_tasks:
                        usage = Usage.model_validate({**user_context.dict(), **extract_usage_info(response_json)})

                        background_tasks.add_task(report_usage_event_task, usage)
                except Exception as e:
                    print(f"Exception occured during usage report {str(e)}")

            # Return processed JSON response
            return Response(
                content=await response.read(),
                status_code=response.status,
                headers=response.headers,
                media_type=content_type,
            )
        except Exception:
            # If JSON parsing fails, fall back to streaming
            return StreamingResponse(
                content=response.content.iter_any(),
                status_code=response.status,
                headers=dict(response.headers),
            )

    # For non-JSON responses, use streaming
    return StreamingResponse(
        content=response.content.iter_any(),
        status_code=response.status,
        headers=dict(response.headers),
    )


def extract_usage_info(response_json: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Extract token cached and predicted counts from JSON response.

    Args:
        response_json: The JSON response from the server
    """

    usage = response_json.get("usage")
    return {
        "input_tokens": int(usage.get("prompt_tokens")),
        "output_tokens": int(usage.get("completion_tokens")),
        "cached_tokens": 0,
    }


@router.post("/{full_path:path}")
async def proxy_request(
    full_path: str,
    request: Request,
    proxy_request: ProxyRequest,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    background_tasks: BackgroundTasks | None = None,
):
    token = credentials.credentials
    if not keys_manager.key_exists(token):
        return Response(status_code=HTTPStatus.UNAUTHORIZED)

    # Get model from request
    model_name = proxy_request.model

    user_context = UserContext(key=token, model_name=model_name)

    # Select server
    server = select_server(user_context, proxy_request.prefer_gpu)
    if not server:
        raise HTTPException(status_code=404, detail=f"No server available for model {user_context.model_name}")

    # Get the original request body
    body = await request.json()

    # Forward the request to the selected server
    async with aiohttp.ClientSession() as session:
        try:
            forwarded_headers = {}
            if config.FORWARD_AUTH:
                headers = dict(request.headers)
                forwarded_headers["authorization"] = headers["authorization"]
            # Forward the request to the selected server
            url = f"{server.url}/{full_path}"
            print(f"forward request to {url}, method: {request.method}, payload: {body}")

            async with session.request(
                method=request.method,
                url=url,
                json=body,
                headers=forwarded_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                # Process and return the response
                return await process_response(response, user_context, background_tasks)
        except Exception as e:
            print("error", e)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")
