import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "set_model_prices.py"
    spec = importlib.util.spec_from_file_location("set_model_prices", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


set_model_prices = _load_module()


def _model(model_id, input_price, output_price, **extra):
    pricing = {
        "price_per_million_input_tokens": input_price,
        "price_per_million_output_tokens": output_price,
        **extra,
    }
    return {"id": model_id, "pricing": {"text": pricing}}


def test_apply_prices_updates_targets_and_preserves_everything_else():
    models = [
        _model("qwen3.8-27b", 0.15, 0.50, price_per_million_cached_input_tokens=0.03),
        _model("glm-5.3", 1.40, 4.40),
    ]
    changed = set_model_prices.apply_prices(models, {"qwen3.8-27b": {"input": 0.18, "output": 1.50}})

    assert changed == ["qwen3.8-27b"]
    text = models[0]["pricing"]["text"]
    assert text["price_per_million_input_tokens"] == 0.18
    assert text["price_per_million_output_tokens"] == 1.50
    assert text["price_per_million_cached_input_tokens"] == 0.03
    assert models[1]["pricing"]["text"] == {
        "price_per_million_input_tokens": 1.40,
        "price_per_million_output_tokens": 4.40,
    }


def test_apply_prices_matches_ids_case_insensitively():
    models = [_model("Qwen3.8-27B-TEE", 0.15, 0.50)]
    changed = set_model_prices.apply_prices(models, {"qwen3.8-27b-tee": {"input": 0.18, "output": 1.50}})
    assert changed == ["Qwen3.8-27B-TEE"]
    assert models[0]["pricing"]["text"]["price_per_million_output_tokens"] == 1.50


def test_apply_prices_reports_no_change_when_already_at_target():
    models = [_model("qwen3.5-122b-a10b", 0.28, 2.10)]
    assert set_model_prices.apply_prices(models, {"qwen3.5-122b-a10b": {"input": 0.28, "output": 2.10}}) == []


def test_apply_prices_rejects_missing_model():
    with pytest.raises(ValueError, match="not found"):
        set_model_prices.apply_prices([], {"qwen3.8-27b": {"input": 0.18, "output": 1.50}})


def test_apply_prices_rejects_model_without_text_pricing():
    with pytest.raises(TypeError, match="no text pricing"):
        set_model_prices.apply_prices(
            [{"id": "bge-m3", "pricing": {"embedding": {}}}],
            {"bge-m3": {"input": 0.01, "output": 0.0}},
        )


def test_default_table_keeps_tee_price_equal_to_the_non_tee_endpoint():
    prices = set_model_prices.DEFAULT_PRICES
    assert prices["qwen3.8-27b"] == prices["qwen3.8-27b-tee"]
