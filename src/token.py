from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from libertai.utils.signature import get_token_message

from src.interfaces.account import CreateAccount, TokenMessage
from src.tasks import add_token_task
from src.utils.account import InvalidSignatureError, create_token_from_account

router = APIRouter(tags=["Token service"])


@router.post("/token/message")
async def token_message(message: TokenMessage):

    message_to_sign = get_token_message()
    data = {
        "message": message_to_sign
    }
    return JSONResponse(content=data, status_code=HTTPStatus.OK)


@router.post("/token/create")
async def token_create(account: CreateAccount, background_tasks: BackgroundTasks):
    try:
        data = await create_token_from_account(account)

        background_tasks.add_task(add_token_task, data["token"])
        return JSONResponse(content=data, status_code=HTTPStatus.OK)
    except InvalidSignatureError as e:
        raise HTTPException(status_code=400, detail=str(e))
