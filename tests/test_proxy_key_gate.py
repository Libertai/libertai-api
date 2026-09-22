import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import proxy
from src.api_keys import KeysManager
from src.auth import extract_api_key


def test_extract_api_key_strips_scheme_case_insensitively():
    assert extract_api_key({"authorization": "Bearer abc123"}) == "abc123"
    assert extract_api_key({"authorization": "bearer abc123"}) == "abc123"


def test_extract_api_key_passthrough_without_scheme():
    assert extract_api_key({"authorization": "abc123"}) == "abc123"


def test_extract_api_key_reads_x_api_key():
    assert extract_api_key({"x-api-key": "abc123"}) == "abc123"


def test_extract_api_key_prefers_authorization():
    assert extract_api_key({"authorization": "Bearer from-auth", "x-api-key": "from-header"}) == "from-auth"


def test_extract_api_key_none_when_absent_or_empty():
    assert extract_api_key({}) is None
    assert extract_api_key({"authorization": "Bearer  "}) is None
    assert extract_api_key({"x-api-key": " "}) is None


@pytest.fixture(autouse=True)
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


def _client():
    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app)


def test_blocked_key_gets_403_before_any_upstream_call(monkeypatch):
    # Register the model so the request clears the 404 model-resolution check
    # that runs before the key gate, without ever reaching an upstream server.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    KeysManager().invalid_keys = {"blocked": {"reason": "no_credits", "message": "No credits."}}

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m"},
        headers={"Authorization": "Bearer blocked"},
    )

    assert resp.status_code == 403
    assert resp.json() == {
        "error": {
            "message": "No credits.",
            "type": "invalid_request_error",
            "code": "no_credits",
        }
    }


def test_valid_key_passes_the_gate(monkeypatch):
    # A key in the valid set (and not in the invalid map) must fall through the
    # gate to the forwarding loop. Upstream is stubbed to refuse connections, so
    # reaching the all-servers-failed 503 proves the gate didn't over-block.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _no_loads():
        return {}

    async def _noop(*args, **kwargs):
        return None

    async def _refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_all_loads", _no_loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)

    KeysManager().keys = {"good"}
    KeysManager().invalid_keys = {}

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m"},
        headers={"Authorization": "Bearer good"},
    )

    assert resp.status_code == 503


def test_key_in_both_sets_treated_as_valid(monkeypatch):
    # Defensive overlap case: the valid set wins over the invalid map (the two
    # are disjoint by construction), matching auth/check and the box-side check.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _no_loads():
        return {}

    async def _noop(*args, **kwargs):
        return None

    async def _refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_all_loads", _no_loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)

    KeysManager().keys = {"both"}
    KeysManager().invalid_keys = {"both": {"reason": "no_credits", "message": "No credits."}}

    resp = _client().post(
        "/v1/chat/completions",
        json={"model": "m"},
        headers={"Authorization": "Bearer both"},
    )

    assert resp.status_code == 503  # forwarded, not 403


def test_x_api_key_is_gated_like_a_bearer(monkeypatch):
    # Anthropic SDKs authenticate with x-api-key only; a blocked one must get the
    # same reason response a bearer would, not the x402 payment flow.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    KeysManager().invalid_keys = {"blocked": {"reason": "no_credits", "message": "No credits."}}

    resp = _client().post("/v1/chat/completions", json={"model": "m"}, headers={"x-api-key": "blocked"})

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "no_credits"


def test_x_api_key_is_forwarded_as_a_bearer(monkeypatch):
    # Boxes authenticate on Authorization, so the gateway must normalize the key
    # onto that header before forwarding.
    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _no_loads():
        return {}

    async def _noop(*args, **kwargs):
        return None

    forwarded = {}

    async def _capture(req, **kwargs):
        forwarded.update(req.headers)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_all_loads", _no_loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _capture)

    KeysManager().keys = {"good"}
    KeysManager().invalid_keys = {}

    resp = _client().post("/v1/chat/completions", json={"model": "m"}, headers={"x-api-key": "good"})

    assert resp.status_code == 503
    assert forwarded["authorization"] == "Bearer good"


def test_no_auth_request_still_reaches_x402_payment_flow(monkeypatch):
    # The key gate must stay mutually exclusive with the x402 branch: a request
    # carrying no API key at all gets the 402 payment response, not a key error.
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

    resp = _client().post("/v1/chat/completions", json={"model": "m"})

    assert resp.status_code == 402
    assert resp.json() == {"x402": True}


def _stub_forwarding(monkeypatch, load_sequence, models=None):
    """Stub the upstream so a request reaching the forwarding loop fails over to
    the all-servers-failed 503. Returns the list of send calls (empty when the
    gate rejected before any upstream attempt). load_sequence feeds successive
    get_all_loads() snapshots; the last value repeats once exhausted."""
    monkeypatch.setattr(proxy.config, "MODELS", models or {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    snapshots = list(load_sequence)

    async def _loads():
        if len(snapshots) > 1:
            return snapshots.pop(0)
        return snapshots[0]

    async def _noop(*args, **kwargs):
        return None

    sends: list[str] = []

    async def _refuse(req, **kwargs):
        sends.append(str(req.url))
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_all_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)
    return sends


def _post_with_key(key: str | None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _client().post("/v1/chat/completions", json={"model": "m"}, headers=headers)


def test_free_key_at_hard_load_rejected_with_503_before_any_upstream_call(monkeypatch):
    # At the hard threshold a known free-tier key is rejected outright with the
    # OpenAI-shaped body openai-node displays — no upstream call is made.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 50}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert resp.json() == {
        "error": {
            "message": "Model 'm' is currently overloaded with other requests. Try again later.",
            "type": "server_error",
            "code": "model_overloaded",
        }
    }
    assert resp.headers["retry-after"] == "5"
    assert sends == []


def test_paid_key_bypasses_the_gate_at_hard_load(monkeypatch):
    # A paid tier must reach the forwarding loop even when the pool is at hard
    # load — the gate exists to protect paid users from free load, not to block
    # them. Reaching the all-servers-failed 503 proves the gate didn't block.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    _stub_forwarding(monkeypatch, [{"http://up": 50}])
    KeysManager().keys = {"paid"}
    KeysManager().tiers = {"paid": "pro"}

    resp = _post_with_key("paid")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "All servers unavailable for model m"


def test_free_key_under_soft_load_waits_then_proceeds(monkeypatch):
    # Between the soft and hard thresholds a free key waits for the pool to
    # drain below the soft threshold, then falls through to the forwarding loop.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 30}, {"http://up": 0}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    waits: list[float] = []

    async def _no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert waits == [proxy.FREE_GATE_POLL_INTERVAL]
    assert len(sends) == 1


def test_free_key_waits_up_to_max_wait_then_proceeds(monkeypatch):
    # Soft load is a wait, not a wall: while the pool stays above the soft
    # threshold the free key polls until the max wait elapses, then proceeds
    # anyway into the (still-loaded) pool.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 30}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    waits: list[float] = []
    clock = [0.0]

    async def _no_sleep(seconds):
        waits.append(seconds)

    def _fake_monotonic():
        clock[0] += proxy.FREE_GATE_POLL_INTERVAL
        return clock[0]

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)
    monkeypatch.setattr(proxy, "_monotonic", _fake_monotonic)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert len(sends) == 1
    assert len(waits) > 0
    assert clock[0] >= proxy.FREE_GATE_MAX_WAIT


def test_free_key_rejected_at_hard_load_mid_wait(monkeypatch):
    # If load crosses the hard threshold while the request waits in the soft
    # loop, the mid-wait re-check sheds it instead of admitting it.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 30}, {"http://up": 60}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    async def _no_sleep(seconds):
        pass

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_overloaded"
    assert sends == []


def test_free_key_at_exact_soft_load_waits(monkeypatch):
    # The wait condition is >=: a pool sitting exactly at the soft threshold
    # still waits for it to drain before proceeding.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 25}, {"http://up": 0}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    waits: list[float] = []

    async def _no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert waits == [proxy.FREE_GATE_POLL_INTERVAL]
    assert len(sends) == 1


def test_free_key_below_soft_load_proceeds_immediately(monkeypatch):
    # Below the soft threshold a free key goes straight to the forwarding loop
    # without waiting.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 24}])
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    waits: list[float] = []

    async def _no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert waits == []
    assert len(sends) == 1


def test_refreshed_loads_feed_server_sorting_after_wait(monkeypatch):
    # After a wait the gate hands back the refreshed snapshot, and the server
    # sorting below must use it: the server that drained while the request
    # waited is tried first, not the one that was idle on arrival.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(
        monkeypatch,
        [{"http://a": 30, "http://b": 0}, {"http://a": 0, "http://b": 30}],
        models={"m": ["http://a", "http://b"]},
    )
    KeysManager().keys = {"free"}
    KeysManager().tiers = {"free": "free"}

    async def _no_sleep(seconds):
        pass

    clock = [0.0]

    def _fake_monotonic():
        clock[0] += proxy.FREE_GATE_POLL_INTERVAL
        return clock[0]

    monkeypatch.setattr(proxy, "_sleep", _no_sleep)
    monkeypatch.setattr(proxy, "_monotonic", _fake_monotonic)

    resp = _post_with_key("free")

    assert resp.status_code == 503
    assert [url.split("//", 1)[1].split("/", 1)[0] for url in sends] == ["a", "b"]


def test_free_tier_comparison_is_case_insensitive(monkeypatch):
    # The linchpin comparison must not silently fail open on casing drift.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    sends = _stub_forwarding(monkeypatch, [{"http://up": 50}])
    KeysManager().keys = {"oddcase"}
    KeysManager().tiers = {"oddcase": "Free"}

    resp = _post_with_key("oddcase")

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_overloaded"
    assert sends == []


def test_unknown_tier_key_bypasses_the_gate(monkeypatch):
    # A valid key with no tier entry (sync skew) fails open: it reaches the
    # forwarding loop even at hard load rather than being over-shed.
    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    _stub_forwarding(monkeypatch, [{"http://up": 50}])
    KeysManager().keys = {"skew"}
    KeysManager().tiers = {}

    resp = _post_with_key("skew")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "All servers unavailable for model m"


def test_no_auth_x402_request_bypasses_the_gate_at_hard_load(monkeypatch):
    # The admission gate must not touch the x402 branch: a request with no API
    # key gets the 402 payment response even at hard load.
    from fastapi.responses import JSONResponse

    monkeypatch.setattr(proxy.config, "FREE_SOFT_LOAD", 25)
    monkeypatch.setattr(proxy.config, "FREE_HARD_LOAD", 50)
    _stub_forwarding(monkeypatch, [{"http://up": 50}])

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

    resp = _post_with_key(None)

    assert resp.status_code == 402
    assert resp.json() == {"x402": True}
