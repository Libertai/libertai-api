from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from src.interfaces.account import Account, CreateAccount, TokenMessage
from src.utils.account import get_subscription

router = APIRouter(tags=["Account service"])


@router.get("/account/status")
async def account_status():

    address = "0x1b6060cfe8dc7293948c44ffce33f03a79d51e90"
    data = await get_subscription(address)
    return JSONResponse(content=data, status_code=HTTPStatus.OK)


@router.delete("/account")
async def account_delete(
        account: Account,
        background_tasks: BackgroundTasks
):
    try:
        # @todo
        #
        pass
    except Exception:
        raise HTTPException(status_code=503)


@router.get("/account/metrics/{sha1_token}")
async def account_metrics(
        account: Account,
        background_tasks: BackgroundTasks
):
    try:
        # @todo
        #
        pass
    except Exception:
        raise HTTPException(status_code=503)
