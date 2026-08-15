from collections.abc import Mapping
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.api_keys import KeysManager
from src.errors import invalid_key_response

router = APIRouter(tags=["Auth"])
keys_manager = KeysManager()


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """API key from `Authorization: Bearer <key>` or `x-api-key`.

    The Anthropic SDKs only ever send `x-api-key`; OpenAI-shaped clients send the
    bearer. Authorization wins when both are present.
    """
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        auth_header = auth_header[7:]
    return auth_header.strip() or headers.get("x-api-key", "").strip() or None


async def require_api_key(request: Request) -> str:
    token = extract_api_key(request.headers)
    if token is None:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Not authenticated")
    return token


@router.get("/libertai/auth/check")
async def check_auth(token: Annotated[str, Depends(require_api_key)]):
    if keys_manager.key_exists(token):
        return Response(content="OK", status_code=HTTPStatus.OK)
    invalid_info = keys_manager.key_invalid_info(token)
    if invalid_info is not None:
        return invalid_key_response(invalid_info)
    raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
