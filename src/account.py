from http import HTTPStatus

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from libertai_utils.chains.ethereum import is_eth_signature_valid

from src.account_manager import AccountManager
from src.interfaces.account import Account, AccountListResponse, TokenAccount
from src.utils.account import InvalidSignatureError, get_subscription
from src.utils.signature import get_reveal_message

router = APIRouter(tags=["Account service"])
account_manager = AccountManager()


@router.get("/account/{address}/status")
async def account_status(address):
    data = await get_subscription(address)
    if data:
        return JSONResponse(content=data.dict(), status_code=HTTPStatus.OK)
    else:
        raise HTTPException(status_code=404)

@router.delete("/account/{address}")
async def account_delete(background_tasks: BackgroundTasks):
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


@router.get("/account/{address}/list")
async def account_list(
        address, chain: str | None = "BASE",
        reveal_message_signature: str | None = None
) -> AccountListResponse:
    reveal_tokens = False
    if reveal_message_signature is not None:
        reveal_message = get_reveal_message()
        if not is_eth_signature_valid(reveal_message, reveal_message_signature, address):
            raise HTTPException(status_code=401)
        else:
            reveal_tokens = True

    accounts = account_manager.get_accounts_by_owner(address.lower(), reveal_tokens)

    if len(accounts) == 0:
        raise HTTPException(status_code=404)

    if reveal_tokens is False and accounts[0].token != "**hidden**":
        raise HTTPException(status_code=500)

    data = AccountListResponse(
        accounts=accounts,
        reveal_message=get_reveal_message()
    )
    return JSONResponse(content=jsonable_encoder(data), status_code=HTTPStatus.OK)
