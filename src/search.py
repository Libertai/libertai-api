import httpx
from fastapi import APIRouter, Request, Response

from src.config import config
from src.logger import setup_logger

router = APIRouter(tags=["Search"])
logger = setup_logger(__name__)

timeout = httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=5.0)
client = httpx.AsyncClient(timeout=timeout)


async def close_http_client() -> None:
    await client.aclose()


async def _forward(request: Request, path: str) -> Response:
    url = f"{config.SEARCH_SERVICE_URL}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    body = await request.body()

    try:
        upstream = await client.post(url, content=body, headers=headers, params=request.query_params)
    except httpx.TimeoutException:
        return Response(content='{"error":"search service timeout"}', status_code=504, media_type="application/json")
    except httpx.HTTPError as e:
        logger.error(f"Search service error forwarding to {url}: {type(e).__name__}: {e}")
        return Response(content='{"error":"search service unavailable"}', status_code=502, media_type="application/json")

    response_headers = dict(upstream.headers)
    response_headers.pop("content-length", None)
    response_headers.pop("content-encoding", None)
    response_headers.pop("transfer-encoding", None)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


@router.post("/search")
async def search(request: Request) -> Response:
    return await _forward(request, "search")


@router.post("/search/fetch")
async def fetch(request: Request) -> Response:
    return await _forward(request, "fetch")
