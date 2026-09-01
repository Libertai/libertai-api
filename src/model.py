import time
from decimal import Decimal

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.aleph import aleph_service
from src.config import config

router = APIRouter(tags=["Models"])

# Sampling parameters our vLLM backends accept, in OpenRouter's vocabulary
SUPPORTED_SAMPLING_PARAMETERS = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "stop",
    "seed",
    "max_tokens",
    "logit_bias",
]


def _per_token_price(price_per_million_tokens: float) -> str:
    """USD per token as a decimal string (OpenRouter pricing format), e.g. 0.15 -> '0.00000015'."""
    return format(Decimal(str(price_per_million_tokens)) / Decimal(1_000_000), "f")


def _text_metadata(meta: dict) -> tuple[dict, dict | None]:
    capabilities = meta.get("capabilities", {}).get("text", {})
    pricing = meta.get("pricing", {}).get("text")
    return capabilities, pricing


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
    if pricing:
        entry["pricing"] = {
            "prompt": _per_token_price(pricing["price_per_million_input_tokens"]),
            "completion": _per_token_price(pricing["price_per_million_output_tokens"]),
        }
    return entry


def openrouter_model_entry(model_name: str, meta: dict, created: int, thinking: bool = False) -> dict | None:
    """One model in the schema OpenRouter's provider listing endpoint requires.

    Returns None for models without text chat pricing (embeddings, TTS, image, search).
    """
    capabilities, pricing = _text_metadata(meta)
    if not pricing or not capabilities:
        return None

    features = []
    if capabilities.get("function_calling"):
        features.append("tools")
    if thinking:
        features.append("reasoning")

    input_modalities = ["text"]
    if capabilities.get("vision"):
        input_modalities.append("image")

    name = meta.get("name") or model_name
    entry: dict = {
        "id": f"{model_name}-thinking" if thinking else model_name,
        "name": f"{name} (Thinking)" if thinking else name,
        "created": created,
        "input_modalities": input_modalities,
        "output_modalities": ["text"],
        "context_length": capabilities.get("context_window"),
        "pricing": {
            "prompt": _per_token_price(pricing["price_per_million_input_tokens"]),
            "completion": _per_token_price(pricing["price_per_million_output_tokens"]),
        },
        "supported_sampling_parameters": SUPPORTED_SAMPLING_PARAMETERS,
        "supported_features": features,
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
    current_timestamp = int(time.time())

    models_data = []
    for model_name in config.MODELS:
        meta = aleph_service.get_model(model_name)
        models_data.append(openai_model_entry(model_name, meta, current_timestamp))
        if aleph_service.is_reasoning_model(model_name):
            models_data.append(openai_model_entry(model_name, meta, current_timestamp, thinking=True))

    return JSONResponse(content={"object": "list", "data": models_data})


@router.get("/openrouter/models")
async def openrouter_models_list():
    """
    Models listing in the schema OpenRouter requires from providers
    (https://openrouter.ai/docs/guides/community/for-providers).
    Only chat models priced in the Aleph aggregate are listed.
    """
    current_timestamp = int(time.time())

    models_data = []
    for model_name in config.MODELS:
        meta = aleph_service.get_model(model_name)
        if meta is None:
            continue
        entry = openrouter_model_entry(model_name, meta, current_timestamp)
        if entry is None:
            continue
        models_data.append(entry)
        if aleph_service.is_reasoning_model(model_name):
            thinking_entry = openrouter_model_entry(model_name, meta, current_timestamp, thinking=True)
            if thinking_entry is not None:
                models_data.append(thinking_entry)

    return JSONResponse(content={"data": models_data})
