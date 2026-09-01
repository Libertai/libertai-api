from src.model import _per_token_price, openai_model_entry, openrouter_model_entry

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


def test_openai_thinking_variant_id():
    entry = openai_model_entry("glm-5.3", META, created=123, thinking=True)
    assert entry["id"] == "glm-5.3-thinking"


def test_openrouter_entry_schema():
    entry = openrouter_model_entry("glm-5.3", META, created=123)
    assert entry is not None
    assert entry["id"] == "glm-5.3"
    assert entry["name"] == "GLM 5.3"
    assert entry["input_modalities"] == ["text"]
    assert entry["output_modalities"] == ["text"]
    assert entry["context_length"] == 128000
    assert entry["pricing"] == {"prompt": "0.00000015", "completion": "0.0000006"}
    assert "tools" in entry["supported_features"]
    assert "reasoning" not in entry["supported_features"]
    assert entry["hugging_face_id"] == "zai-org/GLM-5.3"
    assert "temperature" in entry["supported_sampling_parameters"]


def test_openrouter_thinking_variant_declares_reasoning():
    entry = openrouter_model_entry("glm-5.3", META, created=123, thinking=True)
    assert entry is not None
    assert entry["id"] == "glm-5.3-thinking"
    assert entry["name"] == "GLM 5.3 (Thinking)"
    assert "reasoning" in entry["supported_features"]


def test_openrouter_entry_skips_non_chat_models():
    assert openrouter_model_entry("bge-m3", EMBEDDING_META, created=123) is None


def test_openrouter_vision_model_lists_image_modality():
    meta = {
        **META,
        "capabilities": {"text": {**META["capabilities"]["text"], "vision": True}},
    }
    entry = openrouter_model_entry("glm-5.3", meta, created=123)
    assert entry is not None
    assert entry["input_modalities"] == ["text", "image"]
