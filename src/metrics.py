"""Prometheus instrumentation: transport middleware + business counters.

/metrics is served by this middleware rather than a FastAPI route so it
never appears in the OpenAPI docs, and the token check runs before any app
code. Labels stay bounded: model is always the canonical resolved name, server
is a configured upstream identity, endpoint maps to a fixed set — no
client-controlled values, so the metric surface is constant regardless of
traffic.
"""

from __future__ import annotations

import hmac
import time

from prometheus_client import Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import config

# Latency is dominated by upstream inference (seconds to minutes), so buckets
# extend well past Prometheus' 10s default tail.
_DURATION_BUCKETS = (0.1, 0.3, 1.2, 5, 15, 60, 300, 600)
_GATE_WAIT_BUCKETS = (0.1, 0.3, 1.2, 5)

# Transport-level (collecting middleware): endpoint/method/status only.
http_requests_total = Counter(
    "libertai_http_requests_total",
    "HTTP requests handled by the proxy, by endpoint/method/status.",
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
    "Chat requests by resolved model and request tier (x402/unknown/free/paid).",
    ["model", "tier"],
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
    "Forwarding attempts that failed over to another server, by model/server.",
    ["model", "server"],
)
upstream_request_duration_seconds = Histogram(
    "libertai_upstream_request_duration_seconds",
    "Time to a successful upstream response (incl. failover), by model.",
    ["model"],
    buckets=_DURATION_BUCKETS,
)

_ENDPOINT_BY_PATH = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/openrouter/v1/chat/completions": "openrouter_chat",
    "/openrouter/chat/completions": "openrouter_chat",
    "/openrouter/v1/completions": "openrouter_completions",
    "/v1/models": "models",
    "/openrouter/models": "models",
    "/libertai/models": "models",
    "/libertai/auth/check": "auth",
    "/aleph-credits": "aleph_credits",
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
            await self._serve_metrics(scope, receive, send)
            return

        endpoint = _endpoint_label(scope["path"])
        method = scope["method"]
        start = time.monotonic()
        # Starlette's exception middleware always emits a response, but default
        # to 0 so an unset status can't masquerade as a real code.
        status = 0

        async def wrapped_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            http_requests_total.labels(endpoint, method, str(status)).inc()
            http_request_duration_seconds.labels(endpoint, method).observe(time.monotonic() - start)

    async def _serve_metrics(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = config.METRICS_TOKEN
        response: Response
        if not token:
            response = JSONResponse(status_code=401, content={"detail": "metrics not configured"})
        elif not hmac.compare_digest(Request(scope).query_params.get("token", ""), token):
            response = JSONResponse(status_code=401, content={"detail": "invalid token"})
        else:
            response = Response(
                content=generate_latest(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        await response(scope, receive, send)
