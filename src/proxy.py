import json
from http import HTTPStatus

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Cookie
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from src.api_keys import KeysManager
from src.config import config
from src.logger import setup_logger

router = APIRouter(tags=["Proxy"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(
    connect=10.0,  # Connection timeout
    read=600.0,    # Read timeout (10 minutes for long inference)
    write=10.0,    # Write timeout (text prompts only)
    pool=10.0      # Pool connection timeout
)
limits = httpx.Limits(
    max_connections=500,           # Max total concurrent connections
    max_keepalive_connections=100  # Max idle connections to keep alive
)
client = httpx.AsyncClient(timeout=timeout, limits=limits)

# Round-robin counter for load balancing
round_robin_counters: dict[str, int] = {}


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

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Clean up headers
    headers.pop("host", None)

    # Get all configured servers for the model
    all_servers = config.MODELS.get(model, [])
    if not all_servers:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No server configured for model {model_name}",
        )

    # Round-robin load balancing: rotate server list based on counter
    if model not in round_robin_counters:
        round_robin_counters[model] = 0

    counter = round_robin_counters[model]
    round_robin_counters[model] = (counter + 1) % len(all_servers)

    # Rotate list for round-robin
    rotated_servers = all_servers[counter:] + all_servers[:counter]

    # Try preferred server first, then round-robin through others
    servers_to_try = []
    if preferred_server and preferred_server in all_servers:
        servers_to_try.append(preferred_server)
        servers_to_try.extend([s for s in rotated_servers if s != preferred_server])
    else:
        servers_to_try = rotated_servers

    last_error = None

    # Try each server with automatic failover
    for attempt, server in enumerate(servers_to_try, 1):
        url = f"{server}/{full_path}"

        try:
            logger.debug(f"Attempt {attempt}/{len(servers_to_try)}: Forwarding to {url}")
            req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
            response = await client.send(req, stream=True)

            # Success! Update the preferred instances map and create the cookie header
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

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException) as e:
            # Connection error - try next server
            logger.warning(f"Connection failed to {url} (attempt {attempt}/{len(servers_to_try)}): {type(e).__name__}: {e}")
            last_error = e
            continue

        except Exception as e:
            # Other errors - log and fail immediately
            logger.error(f"Error forwarding request to {url}: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {str(e)}")

    # All servers failed
    logger.error(f"All {len(servers_to_try)} servers failed for model {model_name}. Last error: {type(last_error).__name__}: {last_error}")
    raise HTTPException(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        detail=f"All servers unavailable for model {model_name}"
    )
