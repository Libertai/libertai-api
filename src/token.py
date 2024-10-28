from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from libertai.utils.signature import get_token_message

from src.interfaces.account import CreateAccount
from src.tasks import add_application_task
from src.utils.account import (
    InvalidSignatureError,
    SubscriptionNotFoundError,
    SubscriptionNotValidError,
    create_token_from_account,
)

router = APIRouter(tags=["Token service"])


@router.get("/token/message")
async def token_message():
    message_to_sign = get_token_message()
    data = {
        "message": message_to_sign
    }
    return JSONResponse(content=data, status_code=HTTPStatus.OK)


@router.post("/token")
async def token_create(
        account: CreateAccount,
        background_tasks: BackgroundTasks
):
    try:
        data = await create_token_from_account(account)
        background_tasks.add_task(add_application_task, data["account"])
        response = {
            "name": data["account"].name,
            "owner": data["account"].owner,
            "token": data["token"]
        }
        return JSONResponse(content=response, status_code=HTTPStatus.OK)
    except (
            InvalidSignatureError,
            SubscriptionNotFoundError,
            SubscriptionNotValidError
    ) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=503)
