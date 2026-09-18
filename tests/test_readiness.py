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
