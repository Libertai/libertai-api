from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api_keys import KeysManager
from src.errors import invalid_key_response

router = APIRouter(tags=["Auth"])
keys_manager = KeysManager()
security = HTTPBearer()


@router.get("/libertai/auth/check")
async def check_auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]):
    token = credentials.credentials
    if keys_manager.key_exists(token):
        return Response(content="OK", status_code=HTTPStatus.OK)
    invalid_info = keys_manager.key_invalid_info(token)
    if invalid_info is not None:
        return invalid_key_response(invalid_info)
    raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
