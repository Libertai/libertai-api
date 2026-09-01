import time
from decimal import Decimal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.aleph import aleph_service
from src.config import config

router = APIRouter(tags=["Models"])

# Stable across requests so OpenRouter's crawler sees one timestamp per deploy
# instead of a new one on every fetch.
_DEPLOY_TIMESTAMP = int(time.time())

# Sampling parameters our vLLM backends accept, as OpenRouter capability
# descriptors (schema_version 2.4).
SUPPORTED_SAMPLING_PARAMETERS = {
    "temperature": {"type": "range", "min": 0, "max": 2},
    "top_p": {"type": "range", "min": 0, "max": 1},
    "top_k": {"type": "integer", "min": 0, "max": 500},
    "min_p": {"type": "range", "min": 0, "max": 1},
    "frequency_penalty": {"type": "range", "min": -2, "max": 2},
    "presence_penalty": {"type": "range", "min": -2, "max": 2},
    "repetition_penalty": {"type": "range", "min": 0.01, "max": 2},
    "stop": {"type": "array", "max_items": 4},
    "seed": {"type": "integer", "min": 0, "max": 2147483647},
    "logit_bias": {"type": "object"},
}


def _per_token_price(price_per_million_tokens: float) -> str:
    """USD per token as a decimal string (OpenRouter pricing format), e.g. 0.15 -> '0.00000015'."""
    return format(Decimal(str(price_per_million_tokens)) / Decimal(1_000_000), "f")


def _text_metadata(meta: dict) -> tuple[dict, dict | None]:
    capabilities = meta.get("capabilities", {}).get("text", {})
    pricing = meta.get("pricing", {}).get("text")
    return capabilities, pricing


def _text_prices(pricing: dict | None) -> tuple[str | None, str | None]:
    """(prompt, completion) per-token prices from the aggregate entry, if fully priced."""
    if not pricing:
        return None, None
    input_price = pricing.get("price_per_million_input_tokens")
    output_price = pricing.get("price_per_million_output_tokens")
    if input_price is None or output_price is None:
        return None, None
    return _per_token_price(input_price), _per_token_price(output_price)


def openai_model_entry(model_name: str, meta: dict | None, created: int, thinking: bool = False) -> dict:
    entry = {
        "id": f"{model_name}-thinking" if thinking else model_name,
        "object": "model",
        "created": created,
        "owned_by": "libertai",
    }
    if meta is None:
        return entry

    capabilities, pricing = _text_metadata(meta)
    context_window = capabilities.get("context_window")
    if context_window:
        entry["context_length"] = context_window
    if meta.get("hf_id"):
        entry["hugging_face_id"] = meta["hf_id"]
    prompt_price, completion_price = _text_prices(pricing)
    if prompt_price is not None:
        entry["pricing"] = {"prompt": prompt_price, "completion": completion_price}
    return entry


def openrouter_model_entry(model_name: str, meta: dict, created: int, thinking: bool = False) -> dict | None:
    """One model in the schema OpenRouter's provider listing endpoint requires.

    https://openrouter.ai/docs/guides/community/for-providers (schema_version 2.4).
    Returns None for models without complete text chat pricing (embeddings, TTS, image, search).
    """
    capabilities, pricing = _text_metadata(meta)
    prompt_price, completion_price = _text_prices(pricing)
    if not capabilities or prompt_price is None:
        return None

    context_window = capabilities.get("context_window")

    supported_parameters = dict(SUPPORTED_SAMPLING_PARAMETERS)
    if context_window:
        supported_parameters["max_tokens"] = {"type": "integer", "min": 1, "max": context_window, "unit": "token"}
    if capabilities.get("function_calling"):
        supported_parameters["tools"] = {"type": "boolean"}
    if thinking:
        supported_parameters["reasoning"] = {"type": "boolean"}

    text_input_modality: dict = {"type": "text"}
    if context_window:
        text_input_modality["supported_inputs"] = {"max_context_length": {"value": context_window, "unit": "token"}}
    text_input_modality["pricing"] = [{"type": "prompt", "unit": "token", "cost_usd": prompt_price}]
    cached_input_price = pricing.get("price_per_million_cached_input_tokens") if pricing else None
    if cached_input_price is not None:
        text_input_modality["pricing"].append(
            {"type": "cached_prompt", "unit": "token", "cost_usd": _per_token_price(cached_input_price)}
        )

    input_modalities = [text_input_modality]
    if capabilities.get("vision"):
        # Image input is tokenized and billed as prompt tokens.
        input_modalities.append(
            {"type": "image", "pricing": [{"type": "prompt", "unit": "token", "cost_usd": prompt_price}]}
        )

    output_modality: dict = {
        "type": "text",
        "streaming": True,
        "supported_parameters": supported_parameters,
        "pricing": [{"type": "completion", "unit": "token", "cost_usd": completion_price}],
    }
    if context_window:
        output_modality["max_length"] = {"value": context_window, "unit": "token"}

    name = meta.get("name") or model_name
    entry: dict = {
        "schema_version": "2.4",
        "id": f"{model_name}-thinking" if thinking else model_name,
        "name": f"{name} (Thinking)" if thinking else name,
        "created": created,
        "input_modalities": input_modalities,
        "output_modalities": [output_modality],
        "compliance": {"zdr": True, "hipaa": False},
    }
    if meta.get("hf_id"):
        entry["hugging_face_id"] = meta["hf_id"]
    return entry


@router.get("/libertai/models")
async def models_list():
    # Get all configured servers
    data = {}
    for model_name, servers in config.MODELS.items():
        data[model_name] = {"servers": servers}
        if aleph_service.is_reasoning_model(model_name):
            data[f"{model_name}-thinking"] = {"servers": servers}

    return JSONResponse(content=data)


@router.get("/v1/models")
async def openai_models_list():
    """
    Returns a list of available models in OpenAI API format, enriched with
    context length, Hugging Face id and USD-per-token pricing when the model
    is priced in the Aleph LTAI_PRICING aggregate.
    """
    models_data = []
    for model_name in config.MODELS:
        meta = aleph_service.get_model(model_name)
        models_data.append(openai_model_entry(model_name, meta, _DEPLOY_TIMESTAMP))
        if aleph_service.is_reasoning_model(model_name):
            models_data.append(openai_model_entry(model_name, meta, _DEPLOY_TIMESTAMP, thinking=True))

    return JSONResponse(content={"object": "list", "data": models_data})


@router.get("/openrouter/models")
async def openrouter_models_list():
    """
    Models listing in the schema OpenRouter requires from providers
    (https://openrouter.ai/docs/guides/community/for-providers).
    Only chat models priced in the Aleph aggregate are listed.
    """
    models_data = []
    for model_name in config.MODELS:
        meta = aleph_service.get_model(model_name)
        if meta is None:
            continue
        entry = openrouter_model_entry(model_name, meta, _DEPLOY_TIMESTAMP)
        if entry is None:
            continue
        models_data.append(entry)
        if aleph_service.is_reasoning_model(model_name):
            models_data.append(openrouter_model_entry(model_name, meta, _DEPLOY_TIMESTAMP, thinking=True))

    return JSONResponse(content={"data": models_data})
