import asyncio
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from src import server


def test_readiness_requires_keys_and_models(monkeypatch):
    monkeypatch.setattr(server.keys_manager, "keys", {"key"})
    monkeypatch.setattr(server.aleph_service, "models_loaded", True)
    assert server._has_authoritative_state()

    monkeypatch.setattr(server.aleph_service, "models_loaded", False)
    assert not server._has_authoritative_state()

    monkeypatch.setattr(server.aleph_service, "models_loaded", True)
    monkeypatch.setattr(server.keys_manager, "keys", set())
    assert not server._has_authoritative_state()


def test_health_starting_payload_reports_progress(monkeypatch):
    monkeypatch.setattr(server, "_ready", False)
    monkeypatch.setattr(server.keys_manager, "keys", {"key"})
    monkeypatch.setattr(server.aleph_service, "models_loaded", False)
    monkeypatch.setattr(server.x402_manager, "prices", {})

    resp = TestClient(server.app).get("/health")

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "starting",
        "keys_loaded": True,
        "models_loaded": False,
        "prices_loaded": False,
    }


def _stub_leader_jobs(monkeypatch, *, models_loaded):
    monkeypatch.setattr(server.leader, "_is_leader", True)
    monkeypatch.setattr(server.keys_manager, "refresh_keys", AsyncMock())
    monkeypatch.setattr(server.aleph_service, "refresh", AsyncMock())
    sync_mock = AsyncMock()
    monkeypatch.setattr(server.aleph_service, "sync_from_redis", sync_mock)
    monkeypatch.setattr(server.aleph_service, "models_loaded", models_loaded)
    monkeypatch.setattr(server.server_health_monitor, "check_all_servers", AsyncMock())
    monkeypatch.setattr(server.x402_manager, "refresh_prices", AsyncMock())
    return sync_mock


def test_leader_falls_back_to_redis_when_refresh_did_not_load_models(monkeypatch):
    sync_mock = _stub_leader_jobs(monkeypatch, models_loaded=False)
    asyncio.run(server._refresh_state())
    sync_mock.assert_awaited_once()


def test_leader_skips_redis_fallback_when_refresh_loaded_models(monkeypatch):
    sync_mock = _stub_leader_jobs(monkeypatch, models_loaded=True)
    asyncio.run(server._refresh_state())
    sync_mock.assert_not_awaited()
