import logging

import pytest

from src.config import _Config


@pytest.fixture
def _capture_config_logs(monkeypatch, caplog):
    """src.config logs via logging.getLogger(__name__) with propagation on, but
    if it ever switched to setup_logger (propagate=False) caplog capture would
    depend on the pytest version — force propagation and level regardless."""
    monkeypatch.setattr(logging.getLogger("src.config"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="src.config"):
        yield caplog


def test_free_thresholds_default_when_env_unset(monkeypatch):
    # Blank (not delenv) so a developer's local .env can't re-set the value via
    # load_dotenv; _int_env treats blank like unset.
    monkeypatch.setenv("FREE_SOFT_LOAD", "")
    monkeypatch.setenv("FREE_HARD_LOAD", "")
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


def test_free_thresholds_fail_safe_when_soft_exceeds_hard(_capture_config_logs, monkeypatch):
    # SOFT > HARD disables the wait phase but must not crash config load; the
    # operator should get a warning.
    monkeypatch.setenv("FREE_SOFT_LOAD", "100")
    monkeypatch.setenv("FREE_HARD_LOAD", "50")
    cfg = _Config()
    assert cfg.FREE_SOFT_LOAD == 100
    assert cfg.FREE_HARD_LOAD == 50
    assert any("FREE_SOFT_LOAD" in record.message for record in _capture_config_logs.records)


def test_free_thresholds_warn_when_soft_non_positive(_capture_config_logs, monkeypatch):
    # SOFT <= 0 forces the full wait even on an idle pool; the invariant check
    # must catch it, not just SOFT > HARD.
    monkeypatch.setenv("FREE_SOFT_LOAD", "0")
    monkeypatch.setenv("FREE_HARD_LOAD", "50")
    cfg = _Config()
    assert cfg.FREE_SOFT_LOAD == 0
    assert any("expected 0 < SOFT <= HARD" in record.message for record in _capture_config_logs.records)


def test_free_thresholds_warn_when_hard_non_positive(_capture_config_logs, monkeypatch):
    # HARD <= 0 rejects every free request; also caught by the invariant check.
    monkeypatch.setenv("FREE_SOFT_LOAD", "25")
    monkeypatch.setenv("FREE_HARD_LOAD", "-1")
    cfg = _Config()
    assert cfg.FREE_HARD_LOAD == -1
    assert any("expected 0 < SOFT <= HARD" in record.message for record in _capture_config_logs.records)


def test_body_size_limit_defaults_and_env(monkeypatch):
    monkeypatch.setenv("MAX_BODY_SIZE_MB", "")
    assert _Config().MAX_BODY_SIZE_MB == 100

    monkeypatch.setenv("MAX_BODY_SIZE_MB", "25")
    assert _Config().MAX_BODY_SIZE_MB == 25
