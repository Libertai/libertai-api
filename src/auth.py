import hashlib
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.account_manager import AccountManager
from src.tasks import call_event_task

router = APIRouter(tags=["Auth service"])

account_manager = AccountManager()
security = HTTPBearer()

@router.get("/auth/check")
async def check_auth(
        credentials: Annotated[
            HTTPAuthorizationCredentials,
            Depends(security)
        ], background_tasks: BackgroundTasks
):
    token = credentials.credentials
    sha1_token = hashlib.sha1(token.encode()).hexdigest()
    print("check token exists", sha1_token)
    if account_manager.token_exists(sha1_token):
        background_tasks.add_task(call_event_task, sha1_token)
        return Response(content="OK", status_code=HTTPStatus.OK)
    else:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED)
