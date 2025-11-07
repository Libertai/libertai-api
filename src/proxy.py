import json
from http import HTTPStatus

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Cookie
from fastapi.responses import StreamingResponse
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
client = httpx.AsyncClient(timeout=timeout)


@router.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


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

    # Forward the request to the selected server
    url = f"{server}/{full_path}"

    try:
        req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
        response = await client.send(req, stream=True)
        response.raise_for_status()

        # Update the preferred instances map and create the cookie header
        preferred_instances_map[model_name] = server
        updated_cookie_value = json.dumps(preferred_instances_map)

        # Build the Set-Cookie header string manually
        cookie_header = (
            f"preferred_instances={updated_cookie_value}; Max-Age=1800; Path=/; HttpOnly; Secure; SameSite=Lax"
        )

        # Copy original headers and add the Set-Cookie header
        response_headers = dict(response.headers)
        response_headers["set-cookie"] = cookie_header

        is_streaming_response = response.headers.get("content-type", "") == "text/event-stream"

        if is_streaming_response:

            async def generate_chunks():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                finally:
                    await response.aclose()

            return StreamingResponse(
                content=generate_chunks(),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("Content-Type", ""),
            )
        else:
            response_bytes = await response.aread()
            await response.aclose()

            return Response(
                content=response_bytes,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("Content-Type", ""),
            )

    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")
