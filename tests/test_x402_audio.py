import asyncio

from src.x402 import X402Manager


def test_audio_price_is_char_count_times_rate_charged_exact():
    mgr = X402Manager()
    mgr.prices = {"kokoro": {"is_audio": True, "price_per_million_input_characters": 0.70}}
    body = {"model": "kokoro", "input": "x" * 1_000_000}
    assert asyncio.run(mgr.compute_price("kokoro", body)) == (0.70, "exact")


def test_audio_price_floored_at_min_price(monkeypatch):
    monkeypatch.setattr("src.x402.config.X402_MIN_PRICE", 0.005)
    mgr = X402Manager()
    mgr.prices = {"kokoro": {"is_audio": True, "price_per_million_input_characters": 0.70}}
    assert asyncio.run(mgr.compute_price("kokoro", {"model": "kokoro", "input": "hi"})) == (0.005, "exact")
