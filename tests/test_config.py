from src.config import _Config


def test_free_thresholds_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("FREE_SOFT_LOAD", raising=False)
    monkeypatch.delenv("FREE_HARD_LOAD", raising=False)
    cfg = _Config()
    assert cfg.FREE_SOFT_LOAD == 25
    assert cfg.FREE_HARD_LOAD == 50


def test_free_thresholds_parsed_from_env(monkeypatch):
    monkeypatch.setenv("FREE_SOFT_LOAD", "10")
    monkeypatch.setenv("FREE_HARD_LOAD", "100")
    cfg = _Config()
    assert cfg.FREE_SOFT_LOAD == 10
    assert cfg.FREE_HARD_LOAD == 100


def test_free_thresholds_fall_back_on_invalid_env(monkeypatch):
    monkeypatch.setenv("FREE_SOFT_LOAD", "abc")
    monkeypatch.setenv("FREE_HARD_LOAD", "")
    cfg = _Config()
    assert cfg.FREE_SOFT_LOAD == 25
    assert cfg.FREE_HARD_LOAD == 50
