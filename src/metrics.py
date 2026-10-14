"""Prometheus instrumentation: transport middleware + business counters.

/metrics is served by this middleware rather than a FastAPI route so it
never appears in the OpenAPI docs, and the token check runs before any app
code. Labels stay bounded: model is always the canonical resolved name, server
is a configured upstream identity, endpoint and method map to fixed sets — no
client-controlled values, so the metric surface is constant regardless of
traffic. Failed token attempts are counted only by
libertai_metrics_auth_failures_total and never reach the transport counters;
non-GET/HEAD attempts are answered 405 and counted nowhere, so probing both
kinds of misuse shows up only in the proxy access logs (and the auth-failure
counter) while keeping the scrape path quiet. Setting METRICS_TOKEN to the
literal "public" opts into unauthenticated access; unset still means 401.
"""

from __future__ import annotations

import hmac
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import config
from src.constants import FREE_GATE_MAX_WAIT, FREE_GATE_POLL_INTERVAL

# Latency is dominated by upstream inference (seconds to minutes), so buckets
# extend well past Prometheus' 10s default tail.
_DURATION_BUCKETS = (0.1, 0.3, 1.2, 5, 15, 60, 300, 600)
# The gate checks its deadline before each poll sleep, so a real wait can reach
# FREE_GATE_MAX_WAIT + FREE_GATE_POLL_INTERVAL (~5.5s); the derived tail bucket
# keeps the timed-out case finite rather than in +Inf, with one poll of margin
# for scheduling overshoot. sorted() makes the ascending invariant structural:
# Histogram rejects an unsorted sequence at import.
_GATE_WAIT_BUCKETS = sorted({0.1, 0.3, 1.2, FREE_GATE_MAX_WAIT, FREE_GATE_MAX_WAIT + 2 * FREE_GATE_POLL_INTERVAL})

# Transport-level (collecting middleware): endpoint/method/status only.
http_requests_total = Counter(
    "libertai_http_requests_total",
    'HTTP requests handled by the proxy, by endpoint/method/status (status="0" marks '
    "a request that ended before a response started — unhandled crash or client disconnect).",
    ["endpoint", "method", "status"],
)
http_request_duration_seconds = Histogram(
    "libertai_http_request_duration_seconds",
    "HTTP request latency from receipt to response start, by endpoint/method.",
    ["endpoint", "method"],
    buckets=_DURATION_BUCKETS,
)

# Business-level (collecting in src/proxy.py at the decision points).
requests_total = Counter(
    "libertai_requests_total",
    "Proxy requests past auth/model checks, including gate-rejected ones "
    "(gate_rejections_total is the free-tier subset), by resolved model and tier "
    "(x402/unknown/free/paid); 402 responses surface on the transport counter as "
    'libertai_http_requests_total{status="402"}.',
    ["model", "tier"],
)
metrics_auth_failures_total = Counter(
    "libertai_metrics_auth_failures_total",
    "Rejected attempts to read /metrics (missing, wrong, or malformed token).",
)
gate_rejections_total = Counter(
    "libertai_gate_rejections_total",
    "Free-tier requests rejected at hard load (model_overloaded), by model.",
    ["model"],
)
gate_wait_seconds = Histogram(
    "libertai_gate_wait_seconds",
    "Time free-tier requests spent waiting in the soft-load gate, by model.",
    ["model"],
    buckets=_GATE_WAIT_BUCKETS,
)
upstream_failures_total = Counter(
    "libertai_upstream_failures_total",
    "Any forwarding attempt abandoned in favor of failover (connect errors, 5xx, "
    "tee failures, 401/400 retries), by model/server. Always-abandon paths "
    "(connect/5xx/tee) count on the final attempt too; a final-attempt 401/400 is "
    "forwarded to the client instead, so it is not counted here.",
    ["model", "server"],
)
upstream_request_duration_seconds = Histogram(
    "libertai_upstream_request_duration_seconds",
    "Time to an upstream response (incl. failover), by model. Streamed responses "
    "observe at upstream header arrival, non-streamed after the full body.",
    ["model"],
    buckets=_DURATION_BUCKETS,
)

# Path -> bounded endpoint label. Keep in sync with the app's route table: a
# route missing here still counts, but collapses into "other" (bounded, less
# visible to dashboards).
_ENDPOINT_BY_PATH = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
    "/openrouter/v1/chat/completions": "openrouter_chat",
    "/openrouter/chat/completions": "openrouter_chat",
    "/openrouter/v1/completions": "openrouter_completions",
    "/v1/models": "models",
    "/openrouter/models": "models",
    "/libertai/models": "models",
    "/libertai/auth/check": "auth",
    "/libertai/aleph-credits": "aleph_credits",
    "/search": "search",
    "/search/fetch": "search_fetch",
    "/health": "health",
}


def _endpoint_label(path: str) -> str:
    """Map a request path to a bounded endpoint label; unknown paths fall back."""
    if path in _ENDPOINT_BY_PATH:
        return _ENDPOINT_BY_PATH[path]
    if path.startswith("/v1/images"):
        return "images"
    return "other"


_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"})


def _method_label(method: str) -> str:
    """Map a request method to a bounded label; unknown methods fall back.

    Methods are arbitrary tokens (RFC 9110): uvicorn/h11 accepts anything and
    Traefik forwards it, so an unauthenticated client could mint unbounded
    label values. Each unique raw method would permanently add ~11 series
    (child + 8 buckets + count + sum), hence the whitelist.
    """
    if method in _HTTP_METHODS:
        return method
    return "other"


class MetricsMiddleware:
    """Serve the token-gated /metrics endpoint and collect transport metrics.

    Every request is timed and counted; /metrics itself is answered here (with
    a token check) and excluded from the transport metrics to avoid
    self-scrape noise.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] == "/metrics":
            if scope["method"] not in ("GET", "HEAD"):
                response = JSONResponse(
                    status_code=405,
                    content={"detail": "method not allowed"},
                    headers={"Allow": "GET, HEAD"},
                )
                await response(scope, receive, send)
                return
            await self._serve_metrics(scope, receive, send)
            return

        endpoint = _endpoint_label(scope["path"])
        method = _method_label(scope["method"])
        start = time.monotonic()
        # ExceptionMiddleware only re-raises non-HTTPExceptions; the real 500 is
        # emitted by ServerErrorMiddleware, outermost of the stack, after this
        # middleware's finally has already run and gone through the original send
        # (not wrapped_send). Default to 0 so such crashes record status="0" — a
        # deliberate crash marker the finally below cannot hear about.
        status = 0
        duration_observed = False

        async def wrapped_send(message: Message) -> None:
            nonlocal status, duration_observed
            if message["type"] == "http.response.start":
                status = message["status"]
                # Observe on response start, not completion: streaming responses
                # (chat completions) can stream for minutes, and this histogram
                # documents time to the first response bytes.
                http_request_duration_seconds.labels(endpoint, method).observe(time.monotonic() - start)
                duration_observed = True
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            http_requests_total.labels(endpoint, method, str(status)).inc()
            if not duration_observed:
                http_request_duration_seconds.labels(endpoint, method).observe(time.monotonic() - start)

    async def _serve_metrics(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = config.METRICS_TOKEN
        if token == "public":
            # Explicit open-access opt-in: no credential required or checked.
            response = Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        elif token and _token_matches(Request(scope), token):
            response = Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        else:
            # Uniform 401 body: an unauthenticated prober must not learn whether
            # the endpoint is armed but mistyped, or not configured at all. Both
            # still increment the auth-failure counter for operators.
            metrics_auth_failures_total.inc()
            response = JSONResponse(status_code=401, content={"detail": "invalid token"})
        await response(scope, receive, send)


def _token_matches(request: Request, token: str) -> bool:
    """Constant-time compare of the token from ?token= or an Authorization header.

    hmac.compare_digest raises TypeError on non-ASCII str (a client-controlled
    value must never be able to 500 the endpoint), so both sides are compared
    as UTF-8 bytes. The query param wins when present, even empty — fail-closed
    on mismatch with no fallback: a misconfigured scraper stuck with a wrong
    or empty ?token= keeps 401ing despite a valid Bearer header.
    """
    if "token" in request.query_params:
        provided = request.query_params["token"]
    else:
        authorization = request.headers.get("authorization", "")
        provided = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), token.encode("utf-8"))
