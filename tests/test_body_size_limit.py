import asyncio

from src import server
from src.server import _BodySizeLimitMiddleware


def _scope(content_length: int | None) -> dict:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "headers": headers}


def _run_middleware(scope, monkeypatch, max_mb: int | None, receive_messages=None):
    """Run the middleware against a downstream app that reads its body through
    receive and records it, and return (reached_bodies, sent_messages)."""
    monkeypatch.setattr(server.config, "MAX_BODY_SIZE_MB", max_mb if max_mb is not None else 100)
    reached: list[bytes] = []
    sent: list[dict] = []

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        reached.append(body)

    async def send(message):
        sent.append(message)

    async def receive():
        if receive_messages:
            return receive_messages.pop(0)
        return {"type": "http.disconnect"}

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

    assert reached == [b""]
    assert sent == []


def test_chunked_body_under_cap_is_drained_and_replayed(monkeypatch):
    # Chunked uploads carry no content-length: the middleware drains them with
    # the same cap and replays the chunks so the app reads the same body it
    # would have read without the middleware.
    messages = [
        {"type": "http.request", "body": b"chunk1", "more_body": True},
        {"type": "http.request", "body": b"chunk2", "more_body": False},
    ]
    reached, sent = _run_middleware(_scope(None), monkeypatch, None, messages)

    assert reached == [b"chunk1chunk2"]
    assert sent == []


def test_chunked_body_over_cap_rejected(monkeypatch):
    # The same attacker can bypass the content-length check with chunked
    # framing, so the drain must cap those bodies too.
    messages = [
        {"type": "http.request", "body": b"x" * (60 * 1024 * 1024), "more_body": True},
        {"type": "http.request", "body": b"y" * (60 * 1024 * 1024), "more_body": False},
    ]
    reached, sent = _run_middleware(_scope(None), monkeypatch, None, messages)

    assert reached == []
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_chunked_client_disconnect_rejected(monkeypatch):
    # A client disconnecting mid-drain also rejects: forwarding a truncated
    # body would be worse.
    messages = [{"type": "http.disconnect"}]
    reached, sent = _run_middleware(_scope(None), monkeypatch, None, messages)

    assert reached == []
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_unicode_digit_content_length_is_drained_not_500(monkeypatch):
    # isdigit() also accepts Unicode digits that int() rejects ('¹'); the
    # ASCII-digits-only parse falls back to the drain instead of raising a 500.
    scope = {"type": "http", "headers": [(b"content-length", "¹".encode())]}
    messages = [{"type": "http.request", "body": b"ok", "more_body": False}]
    reached, sent = _run_middleware(scope, monkeypatch, None, messages)

    assert reached == [b"ok"]
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
