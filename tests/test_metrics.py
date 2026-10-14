"""Tests for the Prometheus instrumentation (src/metrics.py + proxy point increments)."""

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from src import proxy
from src.api_keys import KeysManager
from src.config import config
from src.metrics import _endpoint_label
from src.server import app

TOKEN = "test-token"


def _client():
    return TestClient(app)


@pytest.fixture
def _reset_keys_manager():
    """Snapshot and restore KeysManager singleton state around each test."""
    manager = KeysManager()
    saved_keys = manager.keys.copy()
    saved_invalid = manager.invalid_keys.copy()
    saved_tiers = manager.tiers.copy()
    yield
    manager.keys = saved_keys
    manager.invalid_keys = saved_invalid
    manager.tiers = saved_tiers


def _stub_forwarding(monkeypatch, models=None) -> list[str]:
    """Stub the upstream so an admitted request fails over with a connect error.

    Returns the list of send urls attempted (empty when the gate rejected the
    request before any upstream attempt).
    """
    monkeypatch.setattr(proxy.config, "MODELS", models or {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _loads(_model):
        return {"http://up": 50}

    async def _noop(*args, **kwargs):
        return None

    sends: list[str] = []

    async def _refuse(req, **kwargs):
        sends.append(str(req.url))
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)
    return sends


def _gate_thresholds(monkeypatch):
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)


def _post_with_key(key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _client().post("/v1/chat/completions", json={"model": "m"}, headers=headers)


def _scrape(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics", params={"token": TOKEN})
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    return resp.text


def _streamed(status: int, body: bytes) -> httpx.Response:
    """A response whose body is still unread, as client.send(stream=True) returns."""

    async def _chunks():
        yield body

    return httpx.Response(status, content=_chunks(), headers={"Content-Type": "application/json"})


def _metric_value(text: str, prefix: str) -> float:
    """Sample value of the single series matching prefix, or 0.0 when absent."""
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    return float(lines[0].split()[-1]) if lines else 0.0


def _two_server_stub(monkeypatch, first: httpx.Response, second: httpx.Response) -> None:
    """Serve a canned response per upstream host, forcing a-then-b attempt order.

    The request key is unknown, so the gate fails open and no x402 stubs are
    needed; tests that need a known key seed KeysManager themselves.
    """
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://a", "http://b"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)
    monkeypatch.setattr(proxy.random, "sample", lambda urls, n: list(urls))

    async def _loads(_model):
        return {"http://a": 0, "http://b": 0}

    async def _noop(*args, **kwargs):
        return None

    async def _send(req, **kwargs):
        resp = first if req.url.host == "a" else second
        resp.request = req
        return resp

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _send)


def test_metrics_disabled_when_token_unset(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", "")
    assert _client().get("/metrics").status_code == 401
    text = _scrape(monkeypatch)
    sample = [line for line in text.splitlines() if line.startswith("libertai_metrics_auth_failures_total")]
    assert sample and float(sample[-1].split()[-1]) >= 1


def test_metrics_public_mode_allows_open_access(monkeypatch):
    # METRICS_TOKEN=public is the explicit opt-in for unauthenticated access:
    # no credential is required, none is checked, and no auth failure counted.
    monkeypatch.setattr(config, "METRICS_TOKEN", "public")
    before = _metric_value(_scrape(monkeypatch), "libertai_metrics_auth_failures_total")
    monkeypatch.setattr(config, "METRICS_TOKEN", "public")
    client = _client()
    assert client.get("/metrics").status_code == 200
    assert client.get("/metrics?token=anything").status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 200
    after = _metric_value(_scrape(monkeypatch), "libertai_metrics_auth_failures_total")
    assert after == before


def test_metrics_rejects_missing_or_wrong_token(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", params={"token": "wrong"}).status_code == 401


def test_metrics_rejects_non_ascii_token_without_500(monkeypatch):
    # hmac.compare_digest raises TypeError on non-ASCII str; a client-controlled
    # token must 401, never bubble into a 500. The raw URL form is essential:
    # params={"token": "%C3%A9"} percent-encodes the % and reaches the app as
    # the ASCII literal %C3%A9, which would pass even without the bytes guard.
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    resp = _client().get("/metrics?token=%C3%A9")
    assert resp.status_code == 401


def test_metrics_pins_the_encode_guard_with_a_non_ascii_token(monkeypatch):
    # A non-ASCII configured token must also authenticate via its percent-encoded
    # UTF-8 bytes, exercising the .encode() on the stored-token side.
    monkeypatch.setattr(config, "METRICS_TOKEN", "tèst")
    client = _client()
    assert client.get("/metrics?token=t%C3%A8st").status_code == 200
    assert client.get("/metrics?token=%C3%A9").status_code == 401


def test_metrics_counts_failed_token_attempts(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    failed = 0
    for url in ("/metrics", "/metrics?token=wrong", "/metrics?token=%C3%A9"):
        assert client.get(url).status_code == 401
        failed += 1
    text = _scrape(monkeypatch)
    sample = [line for line in text.splitlines() if line.startswith("libertai_metrics_auth_failures_total")]
    # The family renders as a bare line (no labels); earlier tests also failed
    # token attempts, so the value must have grown by at least this test's own.
    assert len(sample) == 1
    assert float(sample[0].split()[-1]) >= failed


def test_unhandled_crash_records_status_zero(monkeypatch):
    # A crash before any response.start makes ServerErrorMiddleware emit its 500
    # after this middleware's finally has run and through its own send, so the
    # request must surface as status="0" (the crash marker), never status="500".
    # Pins the middleware-stack ordering: moving MetricsMiddleware inside
    # ExceptionMiddleware would silently flip these to status="500".
    async def _boom(scope, receive, send):
        raise RuntimeError("boom")

    route = Route("/boom", _boom, methods=["GET"])
    app.router.routes.append(route)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/boom").status_code == 500
    finally:
        app.router.routes.remove(route)

    text = _scrape(monkeypatch)
    crash = [line for line in text.splitlines() if 'endpoint="other",method="GET",status="0"' in line]
    assert crash
    assert 'endpoint="other",method="GET",status="500"' not in text


def test_metrics_accepts_bearer_token(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    assert client.get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.get("/metrics", headers={"Authorization": "Bearer "}).status_code == 401


def test_metrics_query_token_wins_over_bearer_on_mismatch(monkeypatch):
    # A present-but-wrong or empty ?token= beats a valid Bearer header:
    # fail-closed with no fallback, so a scraper stuck with a stale query
    # token keeps 401ing and an operator can see why.
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    for url in ("/metrics?token=wrong", "/metrics?token="):
        assert client.get(url, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 401


def test_metrics_not_exposed_in_openapi_docs():
    resp = _client().get("/openapi.json")
    assert resp.status_code == 200
    assert "/metrics" not in resp.json()["paths"]


def test_endpoint_map_covers_every_registered_route():
    # The proxy endpoints are served by the {full_path} catch-all (invisible in
    # the OpenAPI schema), so only concrete paths are checked: a new route must
    # gain a label or its traffic silently collapses into "other". This would
    # have caught the dead /aleph-credits key, which never matched the route
    # actually mounted at /libertai/aleph-credits.
    concrete = [path for path in app.openapi()["paths"] if "{" not in path]
    assert concrete
    for path in concrete:
        assert _endpoint_label(path) != "other", path


def test_metrics_scrape_lists_all_metric_families(monkeypatch):
    text = _scrape(monkeypatch)
    for family in (
        "libertai_http_requests_total",
        "libertai_http_request_duration_seconds",
        "libertai_requests_total",
        "libertai_gate_rejections_total",
        "libertai_gate_wait_seconds",
        "libertai_upstream_failures_total",
        "libertai_upstream_request_duration_seconds",
    ):
        assert f"# HELP {family}" in text


def test_transport_metrics_count_a_request(monkeypatch):
    # /health without the lifespan running answers 503 "starting".
    assert _client().get("/health").status_code == 503
    text = _scrape(monkeypatch)
    assert 'libertai_http_requests_total{endpoint="health",method="GET",status="503"}' in text


def test_method_label_is_bounded(monkeypatch):
    # Methods are arbitrary tokens (RFC 9110); a made-up one must collapse into
    # the "other" label instead of minting a fresh series.
    resp = _client().request("X-RANDOM", "/health")
    assert resp.status_code in (404, 405, 503)
    text = _scrape(monkeypatch)
    assert 'libertai_http_requests_total{endpoint="health",method="other"' in text
    assert 'method="X-RANDOM"' not in text


def test_metrics_self_scrape_is_not_counted(monkeypatch):
    # /metrics requests are answered by the middleware and must never reach the
    # transport counters (self-scrape noise on our own monitoring path).
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    client = _client()
    before = _scrape(monkeypatch)
    assert client.get("/metrics", params={"token": TOKEN}).status_code == 200
    after = _scrape(monkeypatch)
    before_http = [line for line in before.splitlines() if line.startswith("libertai_http_requests_total")]
    after_http = [line for line in after.splitlines() if line.startswith("libertai_http_requests_total")]
    assert before_http == after_http


def test_metrics_rejects_non_get_methods(monkeypatch):
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    assert _client().post("/metrics", json={}).status_code == 405


def test_metrics_head_request_works(monkeypatch):
    # HEAD is allowed alongside GET; the exposition body is server-suppressed.
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    resp = _client().head("/metrics", params={"token": TOKEN})
    assert resp.status_code == 200


def test_gate_rejection_and_tier_metrics(monkeypatch, _reset_keys_manager):
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_overloaded"
    assert sends == []

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="free"}' in text
    assert 'libertai_gate_rejections_total{model="m"}' in text


def test_admitted_request_counts_upstream_failure(monkeypatch, _reset_keys_manager):
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    KeysManager().keys = {"paid"}
    KeysManager().tiers = {"paid": "pro"}

    resp = _post_with_key("paid")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "All servers unavailable for model m"
    assert len(sends) == 1

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="paid"}' in text
    assert 'libertai_upstream_failures_total{model="m",server="http://up"}' in text


def test_5xx_failover_counts_upstream_failure(monkeypatch):
    _two_server_stub(monkeypatch, _streamed(500, b"boom"), _streamed(200, b'{"ok": true}'))

    prefix = 'libertai_upstream_failures_total{model="m",server="http://a"}'
    before = _metric_value(_scrape(monkeypatch), prefix)
    resp = _post_with_key("nobody")
    after = _metric_value(_scrape(monkeypatch), prefix)

    assert resp.status_code == 200
    assert after - before == 1


def test_401_sync_retry_counts_upstream_failure(monkeypatch, _reset_keys_manager):
    _two_server_stub(monkeypatch, _streamed(401, b"unauthorized"), _streamed(200, b'{"ok": true}'))
    KeysManager().keys = {"sync"}
    KeysManager().tiers = {"sync": "pro"}

    prefix = 'libertai_upstream_failures_total{model="m",server="http://a"}'
    before = _metric_value(_scrape(monkeypatch), prefix)
    resp = _post_with_key("sync")
    after = _metric_value(_scrape(monkeypatch), prefix)

    assert resp.status_code == 200
    assert after - before == 1


def test_context_length_retry_counts_upstream_failure(monkeypatch):
    too_long = (
        b'{"error":{"message":"This model\'s maximum context length is 200000 tokens. However, you '
        b'requested 32768 output tokens and your prompt contains at least 167233 input tokens.",'
        b'"type":"BadRequestError","param":"input_tokens","code":400}}'
    )
    _two_server_stub(monkeypatch, _streamed(400, too_long), _streamed(200, b'{"ok": true}'))

    prefix = 'libertai_upstream_failures_total{model="m",server="http://a"}'
    before = _metric_value(_scrape(monkeypatch), prefix)
    resp = _post_with_key("nobody")
    after = _metric_value(_scrape(monkeypatch), prefix)

    assert resp.status_code == 200
    assert after - before == 1


def test_tee_failure_counts_upstream_failure(monkeypatch):
    sends = _stub_forwarding(monkeypatch)

    async def _boom(server, client):
        raise RuntimeError("tee target down")

    monkeypatch.setattr(proxy.tee, "target", _boom)

    prefix = 'libertai_upstream_failures_total{model="m",server="http://up"}'
    before = _metric_value(_scrape(monkeypatch), prefix)
    resp = _post_with_key("nobody")
    after = _metric_value(_scrape(monkeypatch), prefix)

    assert resp.status_code == 503
    assert sends == []
    assert after - before == 1


def test_unknown_tier_key_fails_open_and_is_admitted(monkeypatch, _reset_keys_manager):
    # A key outside the valid set must fail open (admitted) and be recorded as
    # tier="unknown" — conflating it with "paid" would hide sync skew in metrics.
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    KeysManager().keys = {"paid"}
    KeysManager().tiers = {"paid": "pro"}

    resp = _post_with_key("ghost")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "All servers unavailable for model m"
    assert len(sends) == 1

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="unknown"}' in text


def test_verified_x402_request_records_x402_tier(monkeypatch):
    # A no-api-key request with a verified payment clears the x402 branch and is
    # recorded as tier="x402" (payment-verified, not the 402 response — which is
    # visible only on the transport counter as status=402).
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _loads(_model):
        return {"http://up": 0}

    async def _noop(*args, **kwargs):
        return None

    async def _max_price(model, body_json):
        return 1.0

    async def _requirements(model_name, max_price, resource_url):
        return [{"scheme": "exact"}]

    async def _valid(header, requirements):
        return True

    async def _chunks():
        yield b'{"ok": true}'

    async def _ok(req, **kwargs):
        resp = httpx.Response(200, content=_chunks(), headers={"Content-Type": "application/json"})
        resp.request = req
        return resp

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.x402_manager, "compute_max_price", _max_price)
    monkeypatch.setattr(proxy.x402_manager, "fetch_payment_requirements", _requirements)
    monkeypatch.setattr(proxy.x402_manager, "verify_payment", _valid)
    monkeypatch.setattr(proxy.client, "send", _ok)

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m"},
        headers={"x-payment": "some-payment-proof"},
    )

    assert resp.status_code == 200

    text = _scrape(monkeypatch)
    assert 'libertai_requests_total{model="m",tier="x402"}' in text


def test_unverified_x402_402_surfaces_only_on_transport_counter(monkeypatch):
    # The exclusion half of the documented pairing: a no-key, no-payment
    # request gets the 402 and shows up only on the transport counter —
    # requests_total{tier="x402"} counts verified payments, not payment prompts.
    from fastapi.responses import JSONResponse

    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _max_price(model, body_json):
        return 1.0

    async def _requirements(model_name, max_price, resource_url):
        return [{"scheme": "exact"}]

    monkeypatch.setattr(proxy.x402_manager, "compute_max_price", _max_price)
    monkeypatch.setattr(proxy.x402_manager, "fetch_payment_requirements", _requirements)
    monkeypatch.setattr(
        proxy.x402_manager,
        "build_402_response",
        lambda requirements: JSONResponse(status_code=402, content={"x402": True}),
    )

    tier_line = 'libertai_requests_total{model="m",tier="x402"}'
    transport_line = 'libertai_http_requests_total{endpoint="chat_completions",method="POST",status="402"}'

    before = _scrape(monkeypatch)
    resp = _client().post("/v1/chat/completions", json={"model": "m"})
    after = _scrape(monkeypatch)

    assert resp.status_code == 402
    assert resp.json() == {"x402": True}
    assert _metric_value(after, tier_line) == _metric_value(before, tier_line)
    assert _metric_value(after, transport_line) - _metric_value(before, transport_line) == 1


def test_soft_load_wait_and_upstream_success_metrics(monkeypatch, _reset_keys_manager):
    # A free key between the soft and hard thresholds waits once, then reaches a
    # successful upstream: both the gate-wait and the upstream-time histograms
    # get an observation for this model.
    _gate_thresholds(monkeypatch)
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    loads_seq = iter([{"http://up": 30}, {"http://up": 0}])

    async def _loads(_model):
        try:
            return next(loads_seq)
        except StopIteration:
            return {"http://up": 0}

    async def _noop(*args, **kwargs):
        return None

    async def _no_sleep(seconds):
        pass

    async def _chunks():
        yield b'{"ok": true}'

    async def _ok(req, **kwargs):
        resp = httpx.Response(200, content=_chunks(), headers={"Content-Type": "application/json"})
        resp.request = req
        return resp

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy, "_sleep", _no_sleep)
    monkeypatch.setattr(proxy.client, "send", _ok)

    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    resp = _post_with_key("free")

    assert resp.status_code == 200

    text = _scrape(monkeypatch)
    wait_lines = [line for line in text.splitlines() if line.startswith("libertai_gate_wait_seconds_bucket")]
    assert any('model="m"' in line for line in wait_lines)
    upstream_lines = [
        line for line in text.splitlines() if line.startswith("libertai_upstream_request_duration_seconds_bucket")
    ]
    assert any('model="m"' in line for line in upstream_lines)


def test_streaming_upstream_success_observes_duration(monkeypatch, _reset_keys_manager):
    # text/event-stream forces the streaming branch, which observes the upstream
    # histogram at header arrival (before any body bytes are consumed) — the
    # semantics documented in the metric help text.
    _gate_thresholds(monkeypatch)
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _loads(_model):
        return {"http://up": 0}

    async def _noop(*args, **kwargs):
        return None

    async def _chunks():
        yield b'data: {"ok": true}\n\n'

    async def _ok(req, **kwargs):
        resp = httpx.Response(200, content=_chunks(), headers={"Content-Type": "text/event-stream"})
        resp.request = req
        return resp

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _ok)

    KeysManager().keys = {"paid"}
    KeysManager().tiers = {"paid": "pro"}

    def upstream_count():
        # Diff the current value for this model only: the histogram lives in a
        # shared process registry that other tests also increment, and other
        # models' children may render before "m".
        return _metric_value(_scrape(monkeypatch), 'libertai_upstream_request_duration_seconds_count{model="m"}')

    before = upstream_count()
    resp = _post_with_key("paid")
    after = upstream_count()

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.text == 'data: {"ok": true}\n\n'
    assert after - before == 1


def test_wait_then_hard_reject_observes_gate_wait(monkeypatch, _reset_keys_manager):
    # Load crossing the hard threshold while a free request waits in the soft
    # loop sheds it mid-wait: the gate-wait histogram is observed (the branch
    # free-tier users actually hit under load) before the rejection is counted.
    _gate_thresholds(monkeypatch)
    sends = _stub_forwarding(monkeypatch)
    loads_seq = iter([{"http://up": 30}, {"http://up": 60}])

    async def _loads(_model):
        try:
            return next(loads_seq)
        except StopIteration:
            return {"http://up": 60}

    async def _noop(*args, **kwargs):
        return None

    async def _no_sleep(seconds):
        pass

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy, "_sleep", _no_sleep)

    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_overloaded"
    assert sends == []

    text = _scrape(monkeypatch)
    wait_lines = [line for line in text.splitlines() if line.startswith("libertai_gate_wait_seconds_bucket")]
    assert any('model="m"' in line for line in wait_lines)
    assert 'libertai_gate_rejections_total{model="m"}' in text
