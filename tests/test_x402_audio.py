import asyncio

from src.x402 import X402Manager


def test_audio_max_price_is_char_count_times_rate():
    mgr = X402Manager()
    mgr.prices = {"kokoro": {"is_audio": True, "price_per_million_input_characters": 0.70}}
    body = {"model": "kokoro", "input": "x" * 1_000_000}
    price = asyncio.run(mgr.compute_max_price("kokoro", body))
    assert price == 0.70


def test_audio_max_price_floor():
    mgr = X402Manager()
    mgr.prices = {"kokoro": {"is_audio": True, "price_per_million_input_characters": 0.70}}
    price = asyncio.run(mgr.compute_max_price("kokoro", {"model": "kokoro", "input": "hi"}))
    assert price == 0.0001  # tiny input floored
