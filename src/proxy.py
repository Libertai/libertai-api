import asyncio
import json
import random
import time
import uuid
from http import HTTPStatus

import httpx
from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.aleph import aleph_service
from src.api_keys import KeysManager
from src.auth import extract_api_key
from src.config import config
from src.constants import (
    FREE_GATE_MAX_WAIT,
    FREE_GATE_POLL_INTERVAL,
    FREE_REJECT_LOG_INTERVAL,
)
from src.errors import invalid_key_response, model_overloaded_response
from src.health import server_health_monitor
from src.image_stripping import IMAGE_STRIP_PATHS, strip_images
from src.load_tracker import (
    LEASE_REFRESH_INTERVAL,
    get_all_loads,
)
from src.load_tracker import (
    acquire as load_acquire,
)
from src.load_tracker import (
    release as load_release,
)
from src.logger import setup_logger
from src.ssl_trust import SSL_CONTEXT
from src.thinking import disable_thinking, request_thinking
from src.x402 import x402_manager

router = APIRouter(tags=["Proxy"])

# vLLM refuses an over-long prompt at admission with a 400; the surrounding wording
# changes between versions, so match on the phrase that has stayed constant.
_CONTEXT_LENGTH_MARKER = "maximum context length"


def _is_context_length_error(body: bytes) -> bool:
    """True when a 400 body says the prompt exceeds the server's context window."""
    return _CONTEXT_LENGTH_MARKER in body.decode("utf-8", "replace").lower()


def _pool_load(model: str, loads: dict[str, int]) -> int:
    """Aggregate inflight requests across the model's configured servers."""
    return sum(loads.get(s, 0) for s in set(config.MODELS.get(model, [])))


# Module-level aliases so tests can patch the sleeper/clock for the gate below
# without patching the stdlib modules globally (httpx/anyio internals included).
_sleep = asyncio.sleep
_monotonic = time.monotonic

# Throttle state for _log_rejection: last log time + rejections suppressed since.
_reject_log_state = {"last": 0.0, "suppressed": 0}


def _log_rejection(model_name: str, pool_load: int) -> None:
    """Warning-log a hard-load rejection at most once per interval.

    Shedding is expected behavior under load, so a burst would otherwise produce
    one warning line per request; suppressed rejections are counted into the
    next line.
    """
    state = _reject_log_state
    now = _monotonic()
    if now - state["last"] < FREE_REJECT_LOG_INTERVAL:
        state["suppressed"] += 1
        return
    suppressed = state["suppressed"]
    state["suppressed"] = 0
    state["last"] = now
    more = f" (+{suppressed} more since last log)" if suppressed else ""
    logger.warning(
        f"Free-tier request to '{model_name}' rejected at hard load "
        f"(pool load={pool_load} >= {config.FREE_HARD_LOAD}){more}"
    )


def _reject_overloaded(model_name: str, pool_load: int) -> JSONResponse:
    _log_rejection(model_name, pool_load)
    return model_overloaded_response(model_name)


async def _free_tier_gate(
    model: str,
    model_name: str,
    api_key: str | None,
    loads: dict[str, int],
) -> tuple[dict[str, int], JSONResponse | None]:
    """Admission control for free-tier keys.

    Returns the (possibly refreshed) loads snapshot plus an optional hard-load
    rejection response. Paid tiers bypass the gate entirely; unknown keys
    (sync skew, missing tier entry) fail open rather than over-shed. Free keys
    below the soft threshold proceed immediately; between soft and hard they
    wait (bounded) for the pool to drain; at the hard threshold they are
    rejected with the OpenAI-shaped 503 openai-node actually displays.

    The cap is advisory, not strict: the request's own inflight lease is only
    acquired in the forwarding loop below, so N concurrent free arrivals just
    under the hard threshold all pass at once, and replicas gate independently.
    Bounded overshoot is expected for a tunable heuristic.
    """
    # Membership gating: tiers is metadata only, so a key outside the valid set
    # (possible only with inconsistent backend data) fails open like any other
    # unknown. The tier comparison is normalized because the gate hinges on it:
    # casing drift ("Free") must not silently fail every free-tier key open.
    if api_key is None or not keys_manager.key_exists(api_key):
        return loads, None
    if (keys_manager.tier(api_key) or "").lower() != "free":
        return loads, None

    pool_load = _pool_load(model, loads)
    if pool_load >= config.FREE_HARD_LOAD:
        return loads, _reject_overloaded(model_name, pool_load)

    deadline = _monotonic() + FREE_GATE_MAX_WAIT
    waited = False
    while pool_load >= config.FREE_SOFT_LOAD and _monotonic() < deadline:
        if not waited:
            logger.info(f"Free-tier request to '{model_name}' waiting for pool drain under soft load")
            waited = True
        await _sleep(FREE_GATE_POLL_INTERVAL)
        # Each poll HGETALLs every configured server across all models, not just
        # this model's — acceptable at the current box scale; revisit if wait
        # volumes grow. Waiters hold no lease, so a draining pool admits the
        # whole waiting cohort at once; jittered polls or a waiter cap are the
        # first levers if that pressure shows up.
        loads = await get_all_loads()
        pool_load = _pool_load(model, loads)
        if pool_load >= config.FREE_HARD_LOAD:
            return loads, _reject_overloaded(model_name, pool_load)
    return loads, None


keys_manager = KeysManager()


timeout = httpx.Timeout(
    connect=3.0,  # Connection timeout (fast failover)
    read=600.0,  # Read timeout (10 minutes for long inference)
    write=10.0,  # Write timeout (text prompts only)
    pool=5.0,  # Pool connection timeout
)
limits = httpx.Limits(
    max_connections=500,  # Max total concurrent connections
    max_keepalive_connections=100,  # Max idle connections to keep alive
    # Backends are behind a forward proxy: a new connection costs a CONNECT bounded by the read
    # timeout above, not connect. Idle tunnels are reused rather than re-established every 5s.
    keepalive_expiry=120.0,
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
    needs_body_update = (model != model_name.lower()) or aleph_service.is_reasoning_model(model) or should_strip_images
    if needs_body_update:
        try:
            body_json = json.loads(body)
            body_json["model"] = model
            # Reasoning models: disable thinking by default, enable only with -thinking suffix
            if aleph_service.is_reasoning_model(model):
                if thinking_requested:
                    body_json = request_thinking(model, body_json)
                else:
                    body_json = disable_thinking(model, body_json)
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

    # Conditional auth: if no API key at all, use x402 payment flow
    api_key = extract_api_key(request.headers)
    if api_key:
        # Known-but-blocked key: answer with the reason instead of forwarding
        # to a box that would return a generic 401. Unknown keys still fall
        # through to the box check (avoids api/box sync-skew 401s here).
        # Valid set wins over the invalid map (lists are disjoint by
        # construction; matches auth/check and the box-side check).
        if not keys_manager.key_exists(api_key):
            invalid_info = keys_manager.key_invalid_info(api_key)
            if invalid_info is not None:
                return invalid_key_response(invalid_info)
        # Boxes authenticate on Authorization, so an x-api-key-only client (any
        # Anthropic SDK) needs its key moved onto that header before forwarding.
        headers["authorization"] = f"Bearer {api_key}"
    else:
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

    # Free-tier admission control: wait (bounded) under soft load, reject at hard
    # load. Paid tiers and unknown keys bypass; the possibly-refreshed snapshot
    # keeps the sorting below accurate.
    loads, gate_response = await _free_tier_gate(model, model_name, api_key, loads)
    if gate_response is not None:
        return gate_response

    # Tiered ordering: healthy > capable > unknown. Sort BY LOAD within each tier, not
    # across tiers — otherwise a known-bad server with zero inflight load gets tried
    # first over an actually-healthy server that happens to be busy.
    healthy_set = set(healthy_servers)
    capable_set = set(capable_servers)
    unknown_servers = [s for s in all_servers if s not in healthy_set and s not in capable_set]

    def by_load(urls: list[str]) -> list[str]:
        # Shuffle first: sorted() is stable, so equal-load servers would otherwise always
        # come out in models.json order. Loads tie at 0 whenever traffic doesn't overlap,
        # which pins all baseline traffic to the first configured server.
        return sorted(random.sample(urls, len(urls)), key=lambda s: loads.get(s, 0))

    tiered = [*by_load(healthy_servers), *by_load(capable_servers), *by_load(unknown_servers)]

    # Deduplicate preserving tier order
    seen: set[str] = set()
    servers_to_try: list[str] = []
    for s in tiered:
        if s not in seen:
            seen.add(s)
            servers_to_try.append(s)

    # Cookie stickiness (KV cache locality) — promote to front only if currently in the pool
    # and not a known-bad server; otherwise ignore the cookie and let tier ordering pick first.
    if (
        preferred_server
        and preferred_server in servers_to_try
        and (preferred_server in healthy_set or preferred_server in capable_set)
    ):
        servers_to_try.remove(preferred_server)
        servers_to_try.insert(0, preferred_server)

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

            # Replicas run different --max-model-len, so a prompt one refuses for length
            # can still fit on a larger one. The body is small and the response ends here
            # either way, so read it eagerly to classify.
            bad_request_body: bytes | None = None
            if response.status_code == HTTPStatus.BAD_REQUEST:
                bad_request_body = await response.aread()
                if _is_context_length_error(bad_request_body) and attempt < len(servers_to_try):
                    await response.aclose()
                    logger.warning(
                        f"Context-length rejection from {url} (attempt {attempt}/{len(servers_to_try)}); "
                        f"retrying on another server"
                    )
                    last_error = Exception(f"HTTP 400 context length from {server}")
                    continue

            # Success! Update the preferred instances map and create the cookie header
            preferred_instances_map[model] = server
            updated_cookie_value = json.dumps(preferred_instances_map)

            # Build the Set-Cookie header string manually
            cookie_header = (
                f"preferred_instances={updated_cookie_value}; Max-Age=600; Path=/; HttpOnly; Secure; SameSite=Lax"
            )

            # Copy original headers and add the Set-Cookie header
            response_headers = dict(response.headers)
            response_headers["set-cookie"] = cookie_header

            is_streaming_response = "text/event-stream" in response.headers.get("content-type", "")

            if is_streaming_response:

                async def generate_chunks(_server=server, _rid=request_id, _url=url, _response=response):
                    last_refresh = time.monotonic()
                    try:
                        # aiter_raw (not aiter_bytes) so we forward the body exactly as the
                        # upstream encoded it, matching the Content-Encoding header we pass on.
                        async for chunk in _response.aiter_raw():
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
                        await _response.aclose()
                        await load_release(_server, _rid)

                owned = False  # generator's finally now owns the release
                return StreamingResponse(
                    content=generate_chunks(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("Content-Type", ""),
                )
            elif bad_request_body is not None:
                # aread() already decoded any Content-Encoding, so the headers must not
                # keep claiming the body is still encoded.
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                await response.aclose()
                return Response(
                    content=bad_request_body,
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
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {type(e).__name__}: {e!s}")

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
