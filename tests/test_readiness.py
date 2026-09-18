from src import server


def test_readiness_requires_keys_and_models(monkeypatch):
    monkeypatch.setattr(server.keys_manager, "keys", {"key"}, raising=False)
    monkeypatch.setattr(server.aleph_service, "models", {"glm-5.3": {}}, raising=False)
    assert server._has_authoritative_state()

    monkeypatch.setattr(server.aleph_service, "models", {}, raising=False)
    assert not server._has_authoritative_state()

    monkeypatch.setattr(server.aleph_service, "models", {"glm-5.3": {}}, raising=False)
    monkeypatch.setattr(server.keys_manager, "keys", set(), raising=False)
    assert not server._has_authoritative_state()
