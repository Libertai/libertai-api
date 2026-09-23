import asyncio

from src import server
from src.server import _BodySizeLimitMiddleware


def _scope(content_length: int | None) -> dict:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "headers": headers}


def _run_middleware(scope, monkeypatch, max_mb: int | None):
    """Run the middleware against a downstream app that records whether it was
    reached, and return (reached, sent_messages)."""
    monkeypatch.setattr(server.config, "MAX_BODY_SIZE_MB", max_mb if max_mb is not None else 100)
    reached: list[dict] = []
    sent: list[dict] = []

    async def app(scope, receive, send):
        reached.append(scope)

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    mw = _BodySizeLimitMiddleware(app)
    asyncio.run(mw(scope, receive, send))
    return reached, sent


def test_oversized_body_rejected_before_any_read(monkeypatch):
    # A request claiming 101MB against the 100MB default must get a 413 with the
    # OpenAI-shaped body, and the downstream app must never be reached (the
    # body is never read into memory).
    reached, sent = _run_middleware(_scope(101 * 1024 * 1024), monkeypatch, None)

    assert reached == []
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_boundary_body_passes_through(monkeypatch):
    # The limit is strict >: a body of exactly the cap is accepted.
    reached, sent = _run_middleware(_scope(100 * 1024 * 1024), monkeypatch, None)

    assert len(reached) == 1
    assert sent == []


def test_body_without_content_length_passes_through(monkeypatch):
    # Chunked uploads carry no content-length; the middleware must not reject
    # them (bounded memory is the uvicorn-side concern, not this guard's).
    reached, sent = _run_middleware(_scope(None), monkeypatch, None)

    assert len(reached) == 1
    assert sent == []


def test_non_http_scope_passes_through(monkeypatch):
    reached, sent = _run_middleware({"type": "websocket", "headers": []}, monkeypatch, None)

    assert len(reached) == 1
    assert sent == []


def test_disabled_cap_passes_everything(monkeypatch):
    # 0 or negative disables the cap entirely.
    reached, sent = _run_middleware(_scope(10 * 1024 * 1024 * 1024), monkeypatch, 0)

    assert len(reached) == 1
    assert sent == []
