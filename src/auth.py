from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api_keys import KeysManager

router = APIRouter(tags=["Auth service"])
keys_manager = KeysManager()
security = HTTPBearer()

@router.get("/libertai/auth/check")
async def check_auth(
        credentials: Annotated[
            HTTPAuthorizationCredentials,
            Depends(security)
        ]
):
    token = credentials.credentials
    if keys_manager.key_exists(token):
        return Response(content="OK", status_code=HTTPStatus.OK)
    else:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
