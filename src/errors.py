from http import HTTPStatus

from fastapi.responses import JSONResponse


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
    )
