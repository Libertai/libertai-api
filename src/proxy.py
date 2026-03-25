import json
from collections import defaultdict
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
from src.aleph import aleph_service
from src.x402 import x402_manager

router = APIRouter(tags=["Proxy"])
keys_manager = KeysManager()
security = HTTPBearer()

timeout = httpx.Timeout(
    connect=3.0,  # Connection timeout (fast failover)
    read=600.0,  # Read timeout (10 minutes for long inference)
    write=10.0,  # Write timeout (text prompts only)
    pool=5.0,  # Pool connection timeout
)
limits = httpx.Limits(
    max_connections=500,  # Max total concurrent connections
    max_keepalive_connections=100,  # Max idle connections to keep alive
)
client = httpx.AsyncClient(timeout=timeout, limits=limits)

# In-flight load tracking for least-connections balancing (weighted by request body size in bytes)
inflight_load: defaultdict[str, int] = defaultdict(int)


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

    model = model_name.lower()
    # Resolve model redirections (e.g. deprecated model names)
    model = aleph_service.resolve(model)
    if model != model_name.lower():
        logger.debug(f"Redirected model '{model_name}' -> '{model}'")

    # Handle thinking model variants
    thinking_requested = False
    if model.endswith("-thinking"):
        base_model = model.removesuffix("-thinking")
        # Resolve redirections on the base model too (e.g. old-model-thinking -> new-model)
        resolved_base = aleph_service.resolve(base_model)
        if aleph_service.is_reasoning_model(resolved_base):
            thinking_requested = True
            model = resolved_base
            logger.debug(f"Thinking variant requested for model '{model}'")
        # If base model isn't a reasoning model, let it fall through to 404

    preferred_server: str | None = preferred_instances_map.get(model)
    if model not in config.MODELS or not config.MODELS[model]:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )

    # Get the original request body & headers
    headers = dict(request.headers)
    body = await request.body()

    # Update request body if model changed or needs thinking kwargs
    needs_body_update = (model != model_name.lower()) or aleph_service.is_reasoning_model(model)
    if needs_body_update:
        try:
            body_json = json.loads(body)
            body_json["model"] = model
            # Reasoning models: disable thinking by default, enable only with -thinking suffix
            if aleph_service.is_reasoning_model(model) and not thinking_requested:
                body_json.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
            body = json.dumps(body_json).encode()
            headers["content-length"] = str(len(body))
        except json.JSONDecodeError:
            pass

    # Clean up headers
    headers.pop("host", None)

    # Conditional auth: if no Authorization header, use x402 payment flow
    has_auth = request.headers.get("authorization")
    if not has_auth:
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            body_json = {}

        max_price = await x402_manager.compute_max_price(model, body_json)
        if max_price is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Model '{model_name}' not available for x402 payments",
            )

        resource_url = f"{config.PUBLIC_BASE_URL}/{full_path}" if config.PUBLIC_BASE_URL else str(request.url)

        requirements = await x402_manager.fetch_payment_requirements(model_name, max_price, resource_url)
        if not requirements:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Failed to get payment requirements from facilitator",
            )

        payment_header = request.headers.get("x-payment") or request.headers.get("payment-signature")
        if not payment_header:
            return x402_manager.build_402_response(requirements)

        valid = await x402_manager.verify_payment(payment_header, requirements[0])
        if not valid:
            return x402_manager.build_402_response(requirements)

        # Inject x402 auth headers for downstream
        headers["authorization"] = f"Bearer {config.X402_API_KEY}"
        headers["x-payment"] = payment_header
        headers["x-payment-requirements"] = json.dumps(requirements[0])

    # Three-tier server selection: healthy (200) > capable (202) > all (fallback)
    healthy_servers = server_health_monitor.healthy_model_urls.get(model, [])
    capable_servers = server_health_monitor.capable_model_urls.get(model, [])
    all_servers = config.MODELS.get(model, [])

    if not all_servers:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No server configured for model {model_name}",
        )

    # Build prioritized pool: healthy first, then capable, fall back to all
    if healthy_servers:
        servers_pool = healthy_servers
    elif capable_servers:
        servers_pool = capable_servers
    else:
        servers_pool = all_servers

    # Capture weight for inflight tracking (must be same value for increment and decrement)
    body_weight = len(body)

    # Least-connections load balancing: sort by in-flight load, prefer cookie server for KV cache locality
    servers_to_try = []
    if preferred_server and preferred_server in servers_pool:
        # Cookie server always first when healthy (KV cache locality)
        servers_to_try.append(preferred_server)
        # Remaining servers sorted by inflight load ascending for failover
        remaining = sorted(
            [s for s in servers_pool if s != preferred_server],
            key=lambda s: inflight_load[s],
        )
        servers_to_try.extend(remaining)
    else:
        # No cookie preference — pick least loaded
        servers_to_try = sorted(servers_pool, key=lambda s: inflight_load[s])

    logger.debug(
        f"Load balancing for {model}: servers_to_try={[f'{s}(load={inflight_load[s]})' for s in servers_to_try]}, "
        f"preferred={'yes' if preferred_server and preferred_server in servers_pool else 'no'}"
    )

    last_error = None

    # Try each server with automatic failover
    for attempt, server in enumerate(servers_to_try, 1):
        url = f"{server}/{full_path}"

        incremented = False
        try:
            logger.debug(f"Attempt {attempt}/{len(servers_to_try)}: Forwarding to {url}")
            req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
            inflight_load[server] += body_weight
            incremented = True
            response = await client.send(req, stream=True)

            # Retry on server errors (5xx) — upstream is broken, try next server
            if response.status_code >= 500:
                await response.aclose()
                inflight_load[server] -= body_weight
                incremented = False
                logger.warning(
                    f"Server error {response.status_code} from {url} (attempt {attempt}/{len(servers_to_try)})"
                )
                last_error = Exception(f"HTTP {response.status_code} from {server}")
                continue

            # Success! Update the preferred instances map and create the cookie header
            preferred_instances_map[model] = server
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

                async def generate_chunks(_server=server, _weight=body_weight):
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    finally:
                        await response.aclose()
                        inflight_load[_server] -= _weight

                return StreamingResponse(
                    content=generate_chunks(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("Content-Type", ""),
                )
            else:
                response_bytes = await response.aread()
                await response.aclose()
                inflight_load[server] -= body_weight

                return Response(
                    content=response_bytes,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("Content-Type", ""),
                )

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException) as e:
            if incremented:
                inflight_load[server] -= body_weight
            # Connection error - try next server
            logger.warning(
                f"Connection failed to {url} (attempt {attempt}/{len(servers_to_try)}): {type(e).__name__}: {e}"
            )
            last_error = e
            continue

        except Exception as e:
            if incremented:
                inflight_load[server] -= body_weight
            # Other errors - log and fail immediately
            logger.error(f"Error forwarding request to {url}: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {str(e)}")

    # All servers failed
    logger.error(
        f"All {len(servers_to_try)} servers failed for model {model_name}. Last error: {type(last_error).__name__}: {last_error}"
    )
    raise HTTPException(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail=f"All servers unavailable for model {model_name}"
    )
