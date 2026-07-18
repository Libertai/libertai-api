import asyncio
import json
import time
import uuid
from http import HTTPStatus

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Cookie
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from src.config import config
from src.health import server_health_monitor
from src.image_stripping import IMAGE_STRIP_PATHS, strip_images
from src.load_tracker import (
    LEASE_REFRESH_INTERVAL,
    acquire as load_acquire,
    release as load_release,
    get_all_loads,
)
from src.logger import setup_logger
from src.aleph import aleph_service
from src.ssl_trust import SSL_CONTEXT
from src.x402 import x402_manager
from src.api_keys import KeysManager
from src.errors import invalid_key_response

router = APIRouter(tags=["Proxy"])
security = HTTPBearer()

keys_manager = KeysManager()


def bearer_token(auth_header: str) -> str:
    return auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else auth_header.strip()


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
client = httpx.AsyncClient(timeout=timeout, limits=limits, verify=SSL_CONTEXT)


async def close_http_client() -> None:
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

    # Strip image content for text-only models (avoids upstream errors on non-vision models)
    should_strip_images = full_path in IMAGE_STRIP_PATHS and not aleph_service.is_vision_model(model)

    # Update request body if model changed, needs thinking kwargs, or needs image stripping
    needs_body_update = (
        (model != model_name.lower()) or aleph_service.is_reasoning_model(model) or should_strip_images
    )
    if needs_body_update:
        try:
            body_json = json.loads(body)
            body_json["model"] = model
            # Reasoning models: disable thinking by default, enable only with -thinking suffix
            if aleph_service.is_reasoning_model(model) and not thinking_requested:
                body_json.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
            # Non-vision models: drop any image parts so the upstream doesn't reject the request
            if should_strip_images:
                body_json, stripped = strip_images(full_path, body_json)
                if stripped:
                    logger.debug(f"Stripped image content for non-vision model '{model}' on {full_path}")
            body = json.dumps(body_json).encode()
            headers["content-length"] = str(len(body))
        except json.JSONDecodeError:
            pass

    # Clean up headers
    headers.pop("host", None)

    # Transparent compression passthrough: only let the upstream compress with what the
    # client accepts. If the client sent no Accept-Encoding, force identity so httpx
    # doesn't inject its own (gzip/br) — otherwise we'd forward a body encoded with
    # something the client never asked for. The response is streamed back raw (still
    # encoded) below, with the upstream Content-Encoding header left intact.
    if "accept-encoding" not in headers:
        headers["accept-encoding"] = "identity"

    # Conditional auth: if no Authorization header, use x402 payment flow
    has_auth = request.headers.get("authorization")
    if has_auth:
        # Known-but-blocked key: answer with the reason instead of forwarding
        # to a box that would return a generic 401. Unknown keys still fall
        # through to the box check (avoids api/box sync-skew 401s here).
        invalid_info = keys_manager.key_invalid_info(bearer_token(has_auth))
        if invalid_info is not None:
            return invalid_key_response(invalid_info)
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

    # Build ordered server pool: healthy first, then capable, then remaining
    healthy_servers = server_health_monitor.healthy_model_urls.get(model, [])
    capable_servers = server_health_monitor.capable_model_urls.get(model, [])
    all_servers = config.MODELS.get(model, [])

    if not all_servers:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No server configured for model {model_name}",
        )

    # Snapshot inflight request counts from Redis once for sorting
    loads = await get_all_loads()

    # Tiered ordering: healthy > capable > unknown. Sort BY LOAD within each tier, not
    # across tiers — otherwise a known-bad server with zero inflight load gets tried
    # first over an actually-healthy server that happens to be busy.
    healthy_set = set(healthy_servers)
    capable_set = set(capable_servers)
    unknown_servers = [s for s in all_servers if s not in healthy_set and s not in capable_set]

    def by_load(urls: list[str]) -> list[str]:
        return sorted(urls, key=lambda s: loads.get(s, 0))

    tiered = [*by_load(healthy_servers), *by_load(capable_servers), *by_load(unknown_servers)]

    # Deduplicate preserving tier order
    seen: set[str] = set()
    servers_to_try: list[str] = []
    for s in tiered:
        if s not in seen:
            seen.add(s)
            servers_to_try.append(s)

    # Cookie stickiness (KV cache locality) — promote to front only if currently in the pool.
    # If the cookie points to a known-bad server, ignore it.
    if preferred_server and preferred_server in servers_to_try:
        if preferred_server in healthy_set or preferred_server in capable_set:
            servers_to_try.remove(preferred_server)
            servers_to_try.insert(0, preferred_server)
        # else: cookie server is unknown/bad — let tier ordering pick first

    logger.debug(
        f"Load balancing for {model}: servers_to_try={[f'{s}(load={loads.get(s, 0)})' for s in servers_to_try]}, "
        f"preferred={'yes' if preferred_server and preferred_server in servers_to_try else 'no'}"
    )

    last_error = None

    # Try each server with automatic failover
    for attempt, server in enumerate(servers_to_try, 1):
        url = f"{server}/{full_path}"

        # Release is best-effort (cancelled cleanup, uncancelled non-streaming
        # disconnects, killed process) — the lease deadline is the real leak guard.
        request_id = uuid.uuid4().hex
        owned = False
        try:
            logger.debug(f"Attempt {attempt}/{len(servers_to_try)}: Forwarding to {url}")
            req = client.build_request("POST", url, content=body, headers=headers, params=request.query_params)
            await load_acquire(server, request_id)
            owned = True
            response = await client.send(req, stream=True)

            # Retry on server errors (5xx) — upstream is broken, try next server
            if response.status_code >= 500:
                await response.aclose()
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

            is_streaming_response = "text/event-stream" in response.headers.get("content-type", "")

            if is_streaming_response:

                async def generate_chunks(_server=server, _rid=request_id, _url=url):
                    last_refresh = time.monotonic()
                    try:
                        # aiter_raw (not aiter_bytes) so we forward the body exactly as the
                        # upstream encoded it, matching the Content-Encoding header we pass on.
                        async for chunk in response.aiter_raw():
                            now = time.monotonic()
                            # Refresh the lease so streams outlasting LEASE_TTL stay counted.
                            if now - last_refresh >= LEASE_REFRESH_INTERVAL:
                                await load_acquire(_server, _rid)
                                last_refresh = now
                            yield chunk
                    except asyncio.CancelledError:
                        raise  # client disconnect — normal
                    except Exception as e:
                        # Headers already sent; end the stream instead of raising into ASGI.
                        logger.warning(f"Stream from {_url} interrupted: {type(e).__name__}: {e}")
                    finally:
                        await response.aclose()
                        await load_release(_server, _rid)

                owned = False  # generator's finally now owns the release
                return StreamingResponse(
                    content=generate_chunks(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("Content-Type", ""),
                )
            else:
                # Raw bytes, still encoded — kept consistent with the Content-Encoding header.
                response_bytes = b"".join([chunk async for chunk in response.aiter_raw()])
                await response.aclose()
                return Response(
                    content=response_bytes,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("Content-Type", ""),
                )

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.TimeoutException, httpx.ProxyError) as e:
            # Connection error (incl. upstream HTTP-proxy failures) - try next server
            logger.warning(
                f"Connection failed to {url} (attempt {attempt}/{len(servers_to_try)}): {type(e).__name__}: {e}"
            )
            last_error = e
            continue

        except Exception as e:
            # Other errors - log and fail immediately
            logger.error(f"Error forwarding request to {url}: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {str(e)}")

        finally:
            if owned:
                await load_release(server, request_id)

    # All servers failed
    logger.error(
        f"All {len(servers_to_try)} servers failed for model {model_name}. Last error: {type(last_error).__name__}: {last_error}"
    )
    raise HTTPException(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail=f"All servers unavailable for model {model_name}"
    )
