from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.aleph import aleph_service
from src.config import config
from src.model import _per_token_price, openai_model_entry, openrouter_model_entry, router

META = {
    "id": "glm-5.3",
    "name": "GLM 5.3",
    "hf_id": "zai-org/GLM-5.3",
    "pricing": {"text": {"price_per_million_input_tokens": 0.15, "price_per_million_output_tokens": 0.6}},
    "capabilities": {
        "text": {"tee": True, "vision": False, "reasoning": True, "context_window": 128000, "function_calling": True}
    },
}

EMBEDDING_META = {
    "id": "bge-m3",
    "name": "BGE M3",
    "pricing": {"embedding": {"price_per_million_input_tokens": 0.01}},
    "capabilities": {"embedding": {}},
}


def test_per_token_price_has_no_float_artifacts():
    assert _per_token_price(0.15) == "0.00000015"
    assert _per_token_price(0.6) == "0.0000006"
    assert _per_token_price(3) == "0.000003"


def test_openai_entry_enriched_from_aggregate():
    entry = openai_model_entry("glm-5.3", META, created=123)
    assert entry["id"] == "glm-5.3"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "libertai"
    assert entry["context_length"] == 128000
    assert entry["hugging_face_id"] == "zai-org/GLM-5.3"
    assert entry["pricing"] == {"prompt": "0.00000015", "completion": "0.0000006"}


def test_openai_entry_without_aggregate_metadata_stays_bare():
    entry = openai_model_entry("mystery-model", None, created=123)
    assert entry == {"id": "mystery-model", "object": "model", "created": 123, "owned_by": "libertai"}


def test_openai_entry_with_incomplete_pricing_has_no_pricing_key():
    """Aggregate entries without both price keys must not crash or emit partial pricing."""
    entry = openai_model_entry(
        "glm-5.3",
        {"capabilities": {"text": {"context_window": 128000}}, "pricing": {"text": {"price_per_million_input_tokens": 0.15}}},
        created=123,
    )
    assert "pricing" not in entry


def test_openai_thinking_variant_id():
    entry = openai_model_entry("glm-5.3", META, created=123, thinking=True)
    assert entry["id"] == "glm-5.3-thinking"


def test_openrouter_entry_schema():
    entry = openrouter_model_entry("glm-5.3", META, created=123)
    assert entry is not None
    assert entry["schema_version"] == "2.4"
    assert entry["id"] == "glm-5.3"
    assert entry["name"] == "GLM 5.3"
    assert entry["hugging_face_id"] == "zai-org/GLM-5.3"

    text_input = entry["input_modalities"][0]
    assert len(entry["input_modalities"]) == 1
    assert text_input["type"] == "text"
    assert text_input["supported_inputs"] == {"max_context_length": {"value": 128000, "unit": "token"}}
    assert text_input["pricing"] == [{"type": "prompt", "unit": "token", "cost_usd": "0.00000015"}]

    (output,) = entry["output_modalities"]
    assert output["type"] == "text"
    assert output["streaming"] is True
    assert output["max_length"] == {"value": 128000, "unit": "token"}
    assert output["pricing"] == [{"type": "completion", "unit": "token", "cost_usd": "0.0000006"}]
    assert "tools" in output["supported_parameters"]
    assert "reasoning" not in output["supported_parameters"]
    assert "temperature" in output["supported_parameters"]
    assert entry["compliance"] == {"zdr": True, "hipaa": False}


def test_openrouter_thinking_variant_declares_reasoning():
    entry = openrouter_model_entry("glm-5.3", META, created=123, thinking=True)
    assert entry is not None
    assert entry["id"] == "glm-5.3-thinking"
    assert entry["name"] == "GLM 5.3 (Thinking)"
    output = entry["output_modalities"][0]
    assert "reasoning" in output["supported_parameters"]


def test_openrouter_entry_skips_non_chat_models():
    assert openrouter_model_entry("bge-m3", EMBEDDING_META, created=123) is None


def test_openrouter_entry_skips_incomplete_pricing():
    entry = openrouter_model_entry(
        "glm-5.3",
        {"capabilities": {"text": {"context_window": 128000}}, "pricing": {"text": {"price_per_million_input_tokens": 0.15}}},
        created=123,
    )
    assert entry is None


def test_openrouter_vision_model_lists_image_modality():
    meta = {
        **META,
        "capabilities": {"text": {**META["capabilities"]["text"], "vision": True}},
    }
    entry = openrouter_model_entry("glm-5.3", meta, created=123)
    assert entry is not None
    assert [m["type"] for m in entry["input_modalities"]] == ["text", "image"]
    image_modality = entry["input_modalities"][1]
    assert image_modality["pricing"] == [{"type": "prompt", "unit": "token", "cost_usd": "0.00000015"}]


def test_openrouter_cached_prompt_pricing():
    meta = {
        **META,
        "pricing": {
            "text": {
                "price_per_million_input_tokens": 0.15,
                "price_per_million_output_tokens": 0.6,
                "price_per_million_cached_input_tokens": 0.03,
            }
        },
    }
    entry = openrouter_model_entry("glm-5.3", meta, created=123)
    assert entry is not None
    assert entry["input_modalities"][0]["pricing"] == [
        {"type": "prompt", "unit": "token", "cost_usd": "0.00000015"},
        {"type": "cached_prompt", "unit": "token", "cost_usd": "0.00000003"},
    ]


def test_openrouter_entry_without_context_window_omits_lengths():
    meta = {**META, "capabilities": {"text": {**META["capabilities"]["text"], "context_window": None}}}
    entry = openrouter_model_entry("glm-5.3", meta, created=123)
    assert entry is not None
    assert "supported_inputs" not in entry["input_modalities"][0]
    assert "max_length" not in entry["output_modalities"][0]


# --- Route-level tests ------------------------------------------------------


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


AGGREGATE = {
    "glm-5.3": dict(META),
    "bge-m3": EMBEDDING_META,
}


def test_openai_models_route(monkeypatch):
    saved = config.MODELS
    config.MODELS = {"glm-5.3": ["server1"], "bge-m3": ["server2"]}
    monkeypatch.setattr(aleph_service, "get_model", lambda model: AGGREGATE.get(model), raising=True)
    monkeypatch.setattr(aleph_service, "is_reasoning_model", lambda model: model == "glm-5.3", raising=True)
    try:
        resp = _client().get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        ids = [entry["id"] for entry in data]
        assert ids == ["glm-5.3", "glm-5.3-thinking", "bge-m3"]
        glm = data[0]
        assert glm["pricing"] == {"prompt": "0.00000015", "completion": "0.0000006"}
        bge = data[2]
        assert "pricing" not in bge  # embedding-only model, no text pricing
    finally:
        config.MODELS = saved


def test_openai_models_route_does_not_500_on_partial_aggregate_entry(monkeypatch):
    saved = config.MODELS
    config.MODELS = {"glm-5.3": ["server1"]}
    monkeypatch.setattr(
        aleph_service,
        "get_model",
        lambda model: {"capabilities": {"text": {"context_window": 128000}}, "pricing": {"text": {}}},
        raising=True,
    )
    monkeypatch.setattr(aleph_service, "is_reasoning_model", lambda model: False, raising=True)
    try:
        resp = _client().get("/v1/models")
        assert resp.status_code == 200
        assert "pricing" not in resp.json()["data"][0]
    finally:
        config.MODELS = saved


def test_openrouter_models_route(monkeypatch):
    saved = config.MODELS
    config.MODELS = {"glm-5.3": ["server1"], "bge-m3": ["server2"]}
    monkeypatch.setattr(aleph_service, "get_model", lambda model: AGGREGATE.get(model), raising=True)
    monkeypatch.setattr(aleph_service, "is_reasoning_model", lambda model: model == "glm-5.3", raising=True)
    try:
        resp = _client().get("/openrouter/models")
        assert resp.status_code == 200
        data = resp.json()["data"]
        ids = [entry["id"] for entry in data]
        assert ids == ["glm-5.3", "glm-5.3-thinking"]
        assert data[0]["schema_version"] == "2.4"
    finally:
        config.MODELS = saved


def test_libertai_models_route(monkeypatch):
    saved = config.MODELS
    config.MODELS = {"glm-5.3": ["server1"]}
    monkeypatch.setattr(aleph_service, "is_reasoning_model", lambda model: model == "glm-5.3", raising=True)
    try:
        resp = _client().get("/libertai/models")
        assert resp.status_code == 200
        assert resp.json() == {
            "glm-5.3": {"servers": ["server1"]},
            "glm-5.3-thinking": {"servers": ["server1"]},
        }
    finally:
        config.MODELS = saved
