import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import proxy, search
from src.api_keys import KeysManager

NOT_IN_PLAN = {
    "error": {
        "message": "This model is available on LiberClaw paid plans. Upgrade at https://liberclaw.ai",
        "type": "invalid_request_error",
        "code": "model_not_in_plan",
    }
}

ALLOWLIST = {"liberclaw:free": ["flash", "search/*", "cheap-*"]}


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


@pytest.fixture
def sends(monkeypatch):
    """Stub the upstream so a request reaching the forwarding loop fails over to
    the all-servers-failed 503; returns the list of attempted upstream URLs
    (empty when the allowlist blocked the request)."""
    monkeypatch.setattr(proxy.config, "TIER_MODEL_ALLOWLIST", ALLOWLIST)
    monkeypatch.setattr(
        proxy.config,
        "MODELS",
        {"flash": ["http://up"], "big": ["http://up"], "cheap-mini": ["http://up"], "reasoner": ["http://up"]},
    )
    redirects = {"old-flash": "flash", "old-big": "big", "cheap-legacy": "big"}
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: redirects.get(model, model))
    monkeypatch.setattr(proxy.aleph_service, "is_reasoning_model", lambda model: model in ("reasoner", "flash"))

    async def _no_loads(_model):
        return {}

    async def _noop(*args, **kwargs):
        return None

    attempted: list[str] = []

    async def _refuse(req, **kwargs):
        attempted.append(str(req.url))
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(proxy, "get_model_loads", _no_loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.client, "send", _refuse)

    KeysManager().keys = {"free-key", "paid-key", "skew-key", "odd-key", "space-key"}
    KeysManager().invalid_keys = {}
    KeysManager().tiers = {
        "free-key": "liberclaw:free",
        "paid-key": "liberclaw:pro",
        "odd-key": "LiberClaw:Free",
        "space-key": "liberclaw:free ",
    }
    return attempted


def _post(model: str, key: str | None = "free-key", path: str = "/v1/chat/completions"):
    app = FastAPI()
    app.include_router(proxy.router)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return TestClient(app).post(path, json={"model": model}, headers=headers)


def test_allowed_model_passes(sends):
    resp = _post("flash")

    assert resp.status_code == 503
    assert len(sends) == 1


def test_blocked_model_gets_403_before_any_upstream_call(sends):
    resp = _post("big")

    assert resp.status_code == 403
    assert resp.json() == NOT_IN_PLAN
    assert sends == []


def test_model_match_is_case_insensitive(sends):
    assert _post("FLASH").status_code == 503
    assert _post("BIG").status_code == 403


def test_glob_entry_matches_by_prefix(sends):
    assert _post("cheap-mini").status_code == 503


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/embeddings", "/v1/messages", "/v1/audio/speech"])
def test_every_proxied_path_is_checked(sends, path):
    # The catch-all proxy route serves every inference API shape, so the check
    # must not depend on the path.
    assert _post("big", path=path).status_code == 403


def test_thinking_variant_of_blocked_model_is_blocked(sends):
    resp = _post("reasoner-thinking")

    assert resp.status_code == 403
    assert sends == []


def test_thinking_variant_of_allowed_model_passes(sends):
    assert _post("flash-thinking").status_code == 503


def test_redirect_from_unlisted_alias_to_allowed_model_passes(sends):
    assert _post("old-flash").status_code == 503


def test_redirect_to_blocked_model_is_blocked(sends):
    assert _post("old-big").status_code == 403


def test_glob_does_not_reach_through_a_redirect(sends):
    # "cheap-legacy" matches the "cheap-*" glob by name but is served by "big":
    # the served model decides, so it stays blocked.
    assert _post("cheap-legacy").status_code == 403


def test_listed_alias_redirecting_to_unlisted_model_is_blocked(sends, monkeypatch):
    # Redirects come from the Aleph pricing aggregate: an allowed-looking alias
    # repointed at a bigger model must not hand that model to the tier.
    monkeypatch.setattr(proxy.config, "TIER_MODEL_ALLOWLIST", {"liberclaw:free": ["old-big"]})

    resp = _post("old-big")

    assert resp.status_code == 403
    assert sends == []


def test_paid_tier_passes(sends):
    assert _post("big", key="paid-key").status_code == 503


def test_tier_lookup_is_case_insensitive(sends):
    assert _post("big", key="odd-key").status_code == 403


def test_tier_with_stray_whitespace_is_still_gated(sends):
    assert _post("big", key="space-key").status_code == 403


def test_anthropic_messages_with_x_api_key_only_is_blocked(sends):
    # Anthropic SDKs authenticate with x-api-key alone; that must not slip past.
    app = FastAPI()
    app.include_router(proxy.router)

    resp = TestClient(app).post(
        "/v1/messages",
        json={"model": "big", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "free-key"},
    )

    assert resp.status_code == 403
    assert resp.json() == NOT_IN_PLAN
    assert sends == []


def test_streaming_request_is_blocked_the_same_way(sends):
    app = FastAPI()
    app.include_router(proxy.router)

    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "big", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer free-key"},
    )

    assert resp.status_code == 403
    assert resp.json() == NOT_IN_PLAN
    assert sends == []


def test_key_without_tier_passes(sends):
    assert _post("big", key="skew-key").status_code == 503


def test_unknown_key_passes(sends):
    # Not in the valid set: fails open and lets the box decide on the key.
    KeysManager().tiers["ghost"] = "liberclaw:free"

    assert _post("big", key="ghost").status_code == 503


def test_empty_config_blocks_nothing(sends, monkeypatch):
    monkeypatch.setattr(proxy.config, "TIER_MODEL_ALLOWLIST", {})

    assert _post("big").status_code == 503


def test_unknown_model_still_404s(sends):
    # Plans gate what is served; an unknown model keeps its not-found answer.
    assert _post("nope").status_code == 404


def test_blocked_request_is_logged_without_key_material(sends, monkeypatch, caplog):
    import logging

    # setup_logger sets propagate=False; force it so caplog sees the record.
    monkeypatch.setattr(logging.getLogger("src.tier_allowlist"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="src.tier_allowlist"):
        _post("big")

    messages = [record.message for record in caplog.records if "not in plan" in record.message]
    assert messages
    assert "liberclaw:free" in messages[0] and "'big'" in messages[0]
    assert all("free-key" not in message for message in messages)


def _search(path: str, key: str):
    app = FastAPI()
    app.include_router(search.router)
    return TestClient(app).post(path, json={"query": "q"}, headers={"Authorization": f"Bearer {key}"})


@pytest.fixture
def search_sends(sends, monkeypatch):
    attempted: list[str] = []

    async def _post_upstream(url, **kwargs):
        attempted.append(url)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(search.client, "post", _post_upstream)
    return attempted


@pytest.mark.parametrize("path", ["/search", "/search/fetch"])
def test_search_allowed_by_glob(search_sends, path):
    resp = _search(path, "free-key")

    assert resp.status_code == 200
    assert len(search_sends) == 1


@pytest.mark.parametrize("path", ["/search", "/search/fetch"])
def test_search_blocked_when_not_listed(search_sends, monkeypatch, path):
    monkeypatch.setattr(proxy.config, "TIER_MODEL_ALLOWLIST", {"liberclaw:free": ["flash"]})

    resp = _search(path, "free-key")

    assert resp.status_code == 403
    assert resp.json() == NOT_IN_PLAN
    assert search_sends == []


def test_search_paid_tier_passes(search_sends, monkeypatch):
    monkeypatch.setattr(proxy.config, "TIER_MODEL_ALLOWLIST", {"liberclaw:free": ["flash"]})

    assert _search("/search", "paid-key").status_code == 200
