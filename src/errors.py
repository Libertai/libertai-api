from http import HTTPStatus
from math import ceil

from fastapi.responses import JSONResponse

from src.constants import FREE_GATE_MAX_WAIT


def invalid_key_response(info: dict) -> JSONResponse:
    """OpenAI-shaped 403 for a known-but-unusable API key.

    `error.message` is the shape openai-node actually displays; a plain
    FastAPI `detail` body is dropped by it entirely.
    """
    return JSONResponse(
        status_code=HTTPStatus.FORBIDDEN,
        content={
            "error": {
                "message": info.get("message") or "This API key is currently not usable.",
                "type": "invalid_request_error",
                "code": info.get("reason") or "forbidden",
            }
        },
    )


def client_disconnected_response() -> JSONResponse:
    """400 abort for a client that disconnects mid-drain.

    The response goes nowhere (the client is gone), but fronting proxies' logs
    should not report an oversized body where the real cause is a disconnect.
    """
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST,
        content={
            "error": {
                "message": "Client disconnected during the request.",
                "type": "invalid_request_error",
                "code": "client_disconnected",
            }
        },
    )


def body_too_large_response(max_mb: int) -> JSONResponse:
    """OpenAI-shaped 413 for an oversized request body, rejected before it is read."""
    return JSONResponse(
        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        content={
            "error": {
                "message": f"Request body exceeds the {max_mb}MB limit. Try a smaller payload.",
                "type": "invalid_request_error",
                "code": "request_too_large",
            }
        },
    )


def model_overloaded_response(model_name: str) -> JSONResponse:
    """OpenAI-shaped 503 for a free-tier request rejected at hard load.

    Same shape rationale as `invalid_key_response`: openai-node drops a plain
    `detail` body entirely.
    """
    return JSONResponse(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        content={
            "error": {
                "message": f"Model '{model_name}' is currently overloaded with other requests. Try again later.",
                "type": "server_error",
                "code": "model_overloaded",
            }
        },
        # On the order of the gate's own max wait (src/constants.py): a retry that
        # honors the hint re-enters with roughly the admission chance we'd have
        # granted ourselves, instead of hammering the gate back to back. The gate's
        # worst-case in-gate time is this plus one poll overshoot.
        headers={"retry-after": str(ceil(FREE_GATE_MAX_WAIT))},
    )
