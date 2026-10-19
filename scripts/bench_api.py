#!/usr/bin/env python3
"""Per-phase latency benchmark for LibertAI chat completions.

Runs a fixed number of identical prompts against one or more targets at a given
concurrency and reports the full HTTP flow: TCP connect, TLS handshake, request
upload, wait for response headers, time to first chunk, streaming download time
and inter-chunk gaps. This isolates *where* a gateway adds cost compared with
talking to a model server directly.

httpcore (the HTTP engine under httpx) exposes per-request trace callbacks, so
connection and request events are timestamped inside the actual client stack;
chunk arrival times are recorded as the response streams in. No new
dependencies.

The direct model boxes serve self-signed certificates; like the app
(src/ssl_trust.py), this script loads every certs/*.crt pin from the repo, so
verification stays on. Add --cacert for certificates outside the repo, and
--insecure only as a last resort.

Targets are base URLs; the request path is /v1/chat/completions on all of them.
The direct servers for a model are listed by the public models endpoint:

    curl -s https://api.libertai.io/libertai/models \
        | jq '."deepseek-v4.1-flash".servers'

Usage:
    python scripts/bench_api.py \
        --target api=https://api.libertai.io \
        --target direct=https://<server-from-models-list> \
        --token "$LIBERTAI_API_KEY" \
        --requests 40 --concurrency 4

    # Streaming is the default and reports early closes plus ~60s cut-offs.
    # Compare against --no-stream, and probe keepalive TTL with --idle-gap 60
    # (keep --keepalive above the gap so only the server closes the connection).
    # To keep one client talking for N seconds, use --model-wait N: the model is
    # asked for a long answer, then follow-up turns are sent on the same pooled
    # connection whenever it stops. Probe with a small -n, e.g. -n 1 -x 1. This
    # tests long-lived connections and keepalive reuse, not an idle wait.
    # To measure whether junk traffic hurts legitimate requests, run
    # --unauth-concurrency 20: background clients send unauthenticated requests
    # for the whole measured phase; they are reported separately.

Interpretation: `ttfb` includes connect/TLS/upload; `wait` is the server's own
think time after the request is fully sent; `first_chunk` isolates generation
start from headers; `download` is stream time. Generated chars and the
normalized `gen p50 chars/s` line keep runs with different output lengths
comparable. With --model-wait, `total` is the hold duration, `download` sums
streaming time across turns, `turns` counts follow-ups, and `hold-short` marks
conversations that ended before 90% of the requested window. Phase rows are
first-turn; the hold probe tests connection lifetime and reuse.
The gateway defaults GLM to reasoning_effort=low while a direct
call with no field renders as the backend's "Max", so pin --reasoning-effort
to the same value on every target. A request only counts as successful when it
carried a real completion: a parseable body/SSE with non-empty generated
content plus [DONE]/finish_reason for streams. A 200 with an error object,
empty content, or a truncated stream is reported as a failure. A
`ttl60-suspect` stream ended without a terminator after 50-70s -- a proxy or
server idle timeout rather than a normal completion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import ssl
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import certifi
import httpcore

DEFAULT_PROMPT = "Explain in one short paragraph why low latency matters for AI applications."
CERT_DIR = Path(__file__).resolve().parent.parent / "certs"
DONE_MARKER = b"[DONE]"
FINISH_REASON = re.compile(rb'"finish_reason"\s*:\s*"([^"]+)"')

# (request field, table label). Every field is milliseconds or None.
PHASES = [
    ("connect", "connect (TCP, new only)"),
    ("tls", "TLS handshake (new only)"),
    ("pre_send", "pre-send (start -> write)"),
    ("upload", "upload request"),
    ("wait", "wait (upload -> headers)"),
    ("ttfb", "TTFB (start -> headers)"),
    ("first_chunk", "first chunk (headers -> chunk)"),
    ("ttft", "TTFT (start -> first chunk)"),
    ("download", "download (headers -> last chunk)"),
    ("total", "total (start -> last chunk)"),
    ("max_gap", "max inter-chunk gap"),
]


@dataclass
class Flow:
    """Timestamps collected from httpcore trace events for one request."""

    start: float
    connect_start: float | None = None
    connect_end: float | None = None
    tls_start: float | None = None
    tls_end: float | None = None
    send_start: float | None = None
    send_end: float | None = None
    headers_end: float | None = None
    closed: float | None = None
    chunks: list[float] = field(default_factory=list)


class FlowTracer:
    """Timestamp httpcore trace events for one request (per-request extension)."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    async def __call__(self, event: str, info: dict) -> None:
        # Events are "<logger-prefix>.<name>.started|complete", e.g.
        # "connection.connect_tcp.complete"; suffix matching is prefix-proof.
        now = time.monotonic()
        flow = self.flow
        if event.endswith("connect_tcp.started"):
            flow.connect_start = now
        elif event.endswith("connect_tcp.complete"):
            flow.connect_end = now
        elif event.endswith("start_tls.started"):
            flow.tls_start = now
        elif event.endswith("start_tls.complete"):
            flow.tls_end = now
        elif event.endswith("send_request_headers.started"):
            flow.send_start = now
        elif event.endswith(("send_request_body.complete", "send_request_headers.complete")):
            flow.send_end = now
        elif event.endswith("receive_response_headers.complete"):
            flow.headers_end = now
        elif event.endswith("response_closed.complete"):
            flow.closed = now


@dataclass
class RequestResult:
    index: int
    ok: bool
    valid: bool
    status: int | None
    error: str | None
    detail: str | None
    sse: bool
    turns: int
    new_connections: int
    complete: bool
    cut: bool
    ttl60_suspect: bool
    chunks: int
    bytes: int
    events: int
    content_chars: int
    chars_per_second: float | None
    hold_short: bool
    gaps_over_1s: int
    gaps_over_5s: int
    finish_reason: str | None
    connect: float | None
    tls: float | None
    pre_send: float | None
    upload: float | None
    wait: float | None
    ttfb: float | None
    first_chunk: float | None
    ttft: float | None
    download: float | None
    total: float
    max_gap: float | None


@dataclass
class NoiseStats:
    workers: int
    requests: int = 0
    duration: float = 0.0
    statuses: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)


@dataclass
class PhaseStats:
    n: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


def _delta_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _ssl_context(args: argparse.Namespace) -> ssl.SSLContext:
    if args.insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    context = ssl.create_default_context(cafile=certifi.where())
    for crt in sorted(CERT_DIR.glob("*.crt")):
        context.load_verify_locations(cafile=str(crt))
    for crt in args.cacert or []:
        context.load_verify_locations(cafile=str(crt))
    return context


def _make_pool(args: argparse.Namespace, max_connections: int) -> httpcore.AsyncConnectionPool:
    return httpcore.AsyncConnectionPool(
        ssl_context=_ssl_context(args),
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
        keepalive_expiry=args.keepalive,
    )


def _content_text(value: object) -> str:
    """Text of message/delta content, including multimodal parts (no reasoning)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(part.get("text")) for part in value if isinstance(part, dict))
    return ""


def _content_chars(value: object) -> int:
    """Length of message/delta content, including reasoning and multimodal parts."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_chars(part.get("text")) for part in value if isinstance(part, dict))
    return 0


def _parse_stream(payload: bytes) -> tuple[bool, str | None, int, int, str]:
    """Validate an SSE completion: real generated content, not just a 200.

    Returns (valid, invalid_reason, content_chars, data_events, content_text).
    Error events and keepalive comments do not count as output; content_text is
    the answer only (no reasoning) so it can be fed back as conversation history.
    """
    chars = 0
    events = 0
    error_message = None
    text_parts: list[str] = []
    for line in payload.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        events += 1
        if "error" in event:
            error = event["error"]
            error_message = error.get("message") if isinstance(error, dict) else str(error)
            continue
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            chars += _content_chars(delta.get("content"))
            chars += _content_chars(delta.get("reasoning_content"))
            chars += _content_chars(delta.get("reasoning"))
            text_parts.append(_content_text(delta.get("content")))
            if delta.get("tool_calls"):
                chars += 1
    if chars == 0:
        reason = f"stream error: {error_message}" if error_message else "no generated content in SSE stream"
        return False, reason, 0, events, ""
    return True, None, chars, events, "".join(text_parts)


def _parse_json(payload: bytes) -> tuple[bool, str | None, int, int, str]:
    """Validate a non-streaming completion body: choices and message content."""
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return False, "response body is not valid JSON", 0, 0, ""
    if not isinstance(body, dict):
        return False, "response body is not a JSON object", 0, 0, ""
    if "error" in body:
        error = body["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        return False, f"response error: {message}", 0, 0, ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "response has no choices", 0, 0, ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    chars = _content_chars(message.get("content"))
    chars += _content_chars(message.get("reasoning_content"))
    chars += _content_chars(message.get("reasoning"))
    text = _content_text(message.get("content"))
    if message.get("tool_calls"):
        chars += 1
    if chars == 0:
        return False, "choice has no message content", 0, 0, ""
    return True, None, chars, 0, text


def _hold_prompt(args: argparse.Namespace) -> str:
    """Ask for a long answer that the client can keep extending with follow-ups."""
    if args.model_wait <= 0:
        return args.prompt
    return (
        "Write an extremely long, detailed, continuously expanding answer to the request below. "
        "Do not summarize or conclude, and do not ask whether to continue; when you finish a "
        f"thought, continue with the next detail.\n\n{args.prompt}"
    )


async def _single_turn(
    pool: httpcore.AsyncConnectionPool,
    url: str,
    args: argparse.Namespace,
    index: int,
    messages: list[dict],
) -> tuple[RequestResult, str]:
    flow = Flow(start=time.monotonic())
    streaming = not args.no_stream
    body_json = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": streaming,
    }
    if args.reasoning_effort:
        # Sent to both targets so the gateway's default (low for GLM) cannot
        # silently make the two runs do different amounts of thinking.
        body_json["reasoning_effort"] = args.reasoning_effort
    body = json.dumps(body_json).encode()
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
        # Keep the payload readable so chunk timings are unaffected by decoding.
        "Accept-Encoding": "identity",
        "User-Agent": "libertai-bench/0.1",
    }
    extensions = {
        "trace": FlowTracer(flow),
        "timeout": {"connect": args.timeout, "read": args.timeout, "write": args.timeout, "pool": args.timeout},
    }

    status: int | None = None
    sse = False
    error: str | None = None
    payload = bytearray()
    try:
        async with pool.stream("POST", url, headers=headers, content=body, extensions=extensions) as response:
            status = response.status
            sse = any(
                name.lower() == b"content-type" and b"text/event-stream" in value.lower()
                for name, value in response.headers
            )
            async for chunk in response.aiter_stream():
                flow.chunks.append(time.monotonic())
                payload.extend(chunk)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    end = time.monotonic()
    first = flow.chunks[0] if flow.chunks else None
    last = flow.chunks[-1] if flow.chunks else None
    gaps = [later - earlier for earlier, later in zip(flow.chunks, flow.chunks[1:])]
    finish = FINISH_REASON.search(payload)
    finish_reason = finish.group(1).decode() if finish else None
    http_ok = error is None and status is not None and 200 <= status < 300
    if http_ok:
        valid, invalid_reason, content_chars, events, text = (_parse_stream if streaming else _parse_json)(payload)
    else:
        valid, invalid_reason, content_chars, events, text = False, None, 0, 0, ""
    complete = bool(valid and (not streaming or DONE_MARKER in payload or finish_reason is not None))
    # A stream is cut when it started (HTTP 200) but never reached [DONE] or a
    # finish_reason: clean EOF without a terminator, empty SSE, or a mid-stream
    # error such as a proxy dropping the connection. A request only counts as
    # successful when it carried real generated content and terminated.
    cut = bool(streaming and http_ok and not complete)
    ok = http_ok and valid and complete
    detail = None
    if error is None and not http_ok:
        detail = payload[:200].decode("utf-8", "replace")
    elif error is None and not valid:
        detail = invalid_reason
    elif cut:
        detail = "stream ended before [DONE]/finish_reason"
    total_ms = _delta_ms(flow.start, last if last is not None else (flow.closed or end))
    stream_seconds = (last - first) if first is not None and last is not None and last > first else None
    chars_per_second = content_chars / stream_seconds if ok and stream_seconds else None
    download_ms = _delta_ms(flow.headers_end, last)

    return RequestResult(
        index=index,
        ok=ok,
        valid=valid,
        status=status,
        error=error,
        detail=detail,
        sse=sse,
        turns=1,
        new_connections=1 if flow.connect_start is not None else 0,
        complete=complete,
        cut=cut,
        ttl60_suspect=bool(cut and total_ms is not None and 50.0 <= total_ms <= 70_000.0),
        chunks=len(flow.chunks),
        bytes=len(payload),
        events=events,
        content_chars=content_chars,
        chars_per_second=chars_per_second,
        hold_short=False,
        gaps_over_1s=sum(1 for gap in gaps if gap >= 1.0),
        gaps_over_5s=sum(1 for gap in gaps if gap >= 5.0),
        finish_reason=finish_reason,
        connect=_delta_ms(flow.connect_start, flow.connect_end),
        tls=_delta_ms(flow.tls_start, flow.tls_end),
        pre_send=_delta_ms(flow.start, flow.send_start),
        upload=_delta_ms(flow.send_start, flow.send_end),
        wait=_delta_ms(flow.send_end, flow.headers_end),
        ttfb=_delta_ms(flow.start, flow.headers_end),
        first_chunk=_delta_ms(flow.headers_end, first),
        ttft=_delta_ms(flow.start, first),
        download=download_ms,
        total=total_ms if total_ms is not None else 0.0,
        max_gap=max((gap * 1000.0 for gap in gaps), default=None),
    ), text


_CONTINUE_PROMPT = (
    "Continue exactly where you left off. Do not repeat any previous text and do not conclude; keep writing."
)
_HISTORY_TURNS = 4
_MAX_HOLD_TURNS = 120


def _trim_history(messages: list[dict]) -> None:
    """Keep the original prompt plus the last few exchanges to bound context."""
    limit = 1 + 2 * _HISTORY_TURNS
    if len(messages) > limit:
        messages[:] = [messages[0]] + messages[-2 * _HISTORY_TURNS :]


def _aggregate(index: int, turns: list[RequestResult], started: float, model_wait: float) -> RequestResult:
    """Fold a multi-turn conversation into one logical request result."""
    first, last = turns[0], turns[-1]
    total_ms = (time.monotonic() - started) * 1000.0
    failure = next((turn for turn in turns if not turn.ok), None)
    stream_ms = sum(turn.download for turn in turns if turn.download is not None)
    content_chars = sum(turn.content_chars for turn in turns)
    cut = any(turn.cut for turn in turns)
    return RequestResult(
        index=index,
        ok=failure is None,
        valid=all(turn.valid for turn in turns),
        status=last.status,
        error=failure.error if failure else None,
        detail=failure.detail if failure else None,
        sse=first.sse,
        turns=len(turns),
        new_connections=sum(turn.new_connections for turn in turns),
        complete=all(turn.complete for turn in turns),
        cut=cut,
        ttl60_suspect=any(turn.ttl60_suspect for turn in turns) or bool(cut and 50.0 <= total_ms <= 70_000.0),
        chunks=sum(turn.chunks for turn in turns),
        bytes=sum(turn.bytes for turn in turns),
        events=sum(turn.events for turn in turns),
        content_chars=content_chars,
        chars_per_second=content_chars / (stream_ms / 1000.0) if content_chars and stream_ms else None,
        hold_short=total_ms < model_wait * 900.0,
        gaps_over_1s=sum(turn.gaps_over_1s for turn in turns),
        gaps_over_5s=sum(turn.gaps_over_5s for turn in turns),
        finish_reason=last.finish_reason,
        connect=first.connect,
        tls=first.tls,
        pre_send=first.pre_send,
        upload=first.upload,
        wait=first.wait,
        ttfb=first.ttfb,
        first_chunk=first.first_chunk,
        ttft=first.ttft,
        download=stream_ms if stream_ms else None,
        total=total_ms,
        max_gap=max((turn.max_gap for turn in turns if turn.max_gap is not None), default=None),
    )


async def _conversation(
    pool: httpcore.AsyncConnectionPool,
    url: str,
    args: argparse.Namespace,
    index: int,
    messages: list[dict],
) -> RequestResult:
    """Keep the model talking with follow-up turns on the same client until the window elapses."""
    started = time.monotonic()
    turns: list[RequestResult] = []
    while len(turns) < _MAX_HOLD_TURNS:
        turn, text = await _single_turn(pool, url, args, index, messages)
        turns.append(turn)
        if not turn.ok or not text:
            break
        if time.monotonic() - started >= args.model_wait:
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": _CONTINUE_PROMPT})
        _trim_history(messages)
    return _aggregate(index, turns, started, args.model_wait)


async def _request(
    pool: httpcore.AsyncConnectionPool, url: str, args: argparse.Namespace, index: int
) -> RequestResult:
    messages = [{"role": "user", "content": _hold_prompt(args)}]
    if args.model_wait > 0:
        return await _conversation(pool, url, args, index, messages)
    result, _ = await _single_turn(pool, url, args, index, messages)
    return result


async def _run_target(args: argparse.Namespace, base_url: str) -> tuple[list[RequestResult], float, NoiseStats]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    results: list[RequestResult] = []
    pool = None if args.cold else _make_pool(args, max_connections=args.concurrency)

    async def one(index: int) -> RequestResult:
        if args.cold:
            request_pool = _make_pool(args, max_connections=1)
            try:
                return await _request(request_pool, url, args, index)
            finally:
                await request_pool.aclose()
        return await _request(pool, url, args, index)

    for warmup_index in range(args.warmup):
        warmup = await one(-1 - warmup_index)
        if not warmup.ok:
            print(f"  warmup failed: {_describe_failure(warmup)}", file=sys.stderr)
            if warmup.status in (401, 403):
                raise SystemExit("authentication failed during warmup; check --token")

    noise = NoiseStats(workers=args.unauth_concurrency)
    noise_stop = asyncio.Event()
    noise_pool = None
    noise_tasks: list[asyncio.Task] = []
    if args.unauth_concurrency > 0:
        # Unauthenticated traffic is sent without an Authorization header. It
        # should be rejected before any inference; its only job is to compete
        # for connections, event-loop time, and server capacity.
        noise_body = json.dumps(
            {"model": args.model, "messages": [{"role": "user", "content": args.prompt}], "max_tokens": 1}
        ).encode()
        noise_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "libertai-bench-noise/0.1",
        }
        noise_extensions = {
            "timeout": {"connect": args.timeout, "read": args.timeout, "write": args.timeout, "pool": args.timeout}
        }
        noise_pool = _make_pool(args, max_connections=args.unauth_concurrency)

        async def noise_worker() -> None:
            while not noise_stop.is_set():
                try:
                    response = await noise_pool.request(
                        "POST", url, headers=noise_headers, content=noise_body, extensions=noise_extensions
                    )
                    noise.statuses[response.status] += 1
                except Exception as e:
                    noise.errors[type(e).__name__] += 1
                noise.requests += 1

        noise_tasks = [asyncio.create_task(noise_worker()) for _ in range(args.unauth_concurrency)]

    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(args.requests):
        queue.put_nowait(index)

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if args.idle_gap > 0:
                await asyncio.sleep(args.idle_gap)
            results.append(await one(index))

    started = time.monotonic()
    try:
        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, args.requests))))
    finally:
        wall = time.monotonic() - started
        noise_stop.set()
        if noise_tasks:
            await asyncio.gather(*noise_tasks, return_exceptions=True)
        if noise_pool is not None:
            await noise_pool.aclose()
        if pool is not None:
            await pool.aclose()
    noise.duration = wall
    results.sort(key=lambda result: result.index)
    return results, wall, noise


def _describe_failure(result: RequestResult) -> str:
    if result.error:
        return result.error
    detail = f": {result.detail}" if result.detail else ""
    return f"HTTP {result.status}{detail}"


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def _phase_values(results: list[RequestResult], field_name: str) -> list[float]:
    values = (getattr(result, field_name) for result in results if result.ok)
    return [value for value in values if value is not None]


def _stats(values: list[float]) -> PhaseStats | None:
    if not values:
        return None
    ordered = sorted(values)
    return PhaseStats(
        n=len(ordered),
        min_ms=ordered[0],
        mean_ms=statistics.fmean(ordered),
        p50_ms=_percentile(ordered, 50),
        p90_ms=_percentile(ordered, 90),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1],
    )


def _print_table(results: list[RequestResult]) -> None:
    ok = [result for result in results if result.ok]
    print(f"  {'phase (ms)':<36}{'n':>4}{'min':>10}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}")
    for field_name, label in PHASES:
        stats = _stats(_phase_values(ok, field_name))
        if stats is None:
            continue
        print(
            f"  {label:<36}{stats.n:>4}{stats.min_ms:>10.1f}{stats.mean_ms:>10.1f}"
            f"{stats.p50_ms:>10.1f}{stats.p90_ms:>10.1f}{stats.p99_ms:>10.1f}{stats.max_ms:>10.1f}"
        )


def _print_report(
    label: str,
    url: str,
    results: list[RequestResult],
    wall: float,
    streaming: bool,
    noise: NoiseStats,
    model_wait: float,
) -> None:
    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    total_turns = sum(result.turns for result in results)
    new_connections = sum(result.new_connections for result in results)
    throughput = len(ok) / wall if wall > 0 else 0.0

    print(f"\n=== {label} -- {url}")
    print(
        f"  requests {len(results)}  ok {len(ok)}  failed {len(failed)}  "
        f"wall {wall:.1f}s  throughput {throughput:.2f} ok req/s"
    )
    unit = "turns" if model_wait > 0 else "requests"
    print(f"  connections {new_connections} opened / {total_turns} {unit} ({total_turns - new_connections} reused)")
    if noise.workers:
        rate = noise.requests / noise.duration if noise.duration > 0 else 0.0
        statuses = ", ".join(f"{status} {count}" for status, count in noise.statuses.most_common()) or "none"
        print(
            f"  unauth noise: {noise.workers} clients, {noise.requests} requests in {noise.duration:.1f}s "
            f"({rate:.0f}/s)  statuses: {statuses}"
        )
        if noise.errors:
            print(
                "  unauth noise errors: "
                + ", ".join(f"{error} {count}" for error, count in noise.errors.most_common())
            )
        if any(200 <= status < 300 for status in noise.statuses):
            print("  WARNING: unauthenticated requests were admitted (2xx) - auth may be open")
    if streaming:
        streams = [result for result in results if result.sse]
        if streams:
            complete = sum(1 for result in streams if result.complete)
            cuts = sum(1 for result in streams if result.cut)
            ttl60 = sum(1 for result in streams if result.ttl60_suspect)
            gaps_1s = sum(result.gaps_over_1s for result in streams)
            gaps_5s = sum(result.gaps_over_5s for result in streams)
            print(
                f"  streams: complete {complete}  early-close {cuts}  ttl60-suspect {ttl60}  "
                f"gaps >1s {gaps_1s}  gaps >5s {gaps_5s}"
            )
            print(
                f"  chunks {sum(result.chunks for result in streams)}  "
                f"events {sum(result.events for result in streams)}  "
                f"bytes {sum(result.bytes for result in streams)}"
            )
            reasons = Counter(result.finish_reason for result in streams if result.finish_reason)
            if reasons:
                print("  finish reasons: " + ", ".join(f"{reason} {count}" for reason, count in reasons.most_common()))
            if model_wait > 0:
                held = [result.total / 1000.0 for result in streams]
                met = sum(1 for result in streams if result.ok and not result.hold_short)
                hold_p50 = _percentile(sorted(held), 50) if held else 0.0
                turns_p50 = _percentile(sorted(float(result.turns) for result in streams), 50) if streams else 0.0
                print(
                    f"  hold {model_wait:g}s: met {met}/{len(streams)}  held p50 {hold_p50:.1f}s  "
                    f"turns p50 {turns_p50:.0f}  short {sum(1 for result in streams if result.hold_short)}"
                )
                print("  (hold phases below are first-turn; download sums streaming time across turns)")
    if ok:
        chars = [result.content_chars for result in ok]
        rates = [result.chars_per_second for result in ok if result.chars_per_second]
        suffix = f"  gen p50 {_percentile(sorted(rates), 50):.0f} chars/s" if rates else ""
        print(f"  output: {sum(chars)} chars (avg {statistics.fmean(chars):.0f}/request){suffix}")
    _print_table(results)
    if failed:
        print("  failures:")
        counts = Counter(_describe_failure(result) for result in failed)
        for sample, count in counts.most_common(5):
            print(f"    [{count}x] {sample[:160]}")


def _print_comparison(reports: dict[str, list[RequestResult]]) -> None:
    if len(reports) < 2:
        return
    labels = list(reports)
    print("\n=== Comparison (p50 ms, ok requests only)")
    print(f"  {'phase':<36}" + "".join(f"{label:>14}" for label in labels))
    for field_name, label in PHASES:
        cells = []
        for target in labels:
            stats = _stats(_phase_values(reports[target], field_name))
            cells.append(f"{stats.p50_ms:>14.1f}" if stats else f"{'-':>14}")
        print(f"  {label:<36}" + "".join(cells))
    print(f"  {'ok requests':<36}" + "".join(f"{sum(r.ok for r in reports[t]):>14}" for t in labels))


def _write_json(
    path: Path,
    args: argparse.Namespace,
    urls: dict[str, str],
    walls: dict[str, float],
    reports: dict[str, list[RequestResult]],
    noises: dict[str, NoiseStats],
) -> None:
    data = {
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "unauth_concurrency": args.unauth_concurrency,
        "stream": not args.no_stream,
        "cold": args.cold,
        "idle_gap": args.idle_gap,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "prompt": args.prompt,
        "model_wait": args.model_wait,
        "targets": {
            label: {
                "url": urls[label],
                "wall_seconds": walls[label],
                "noise": {
                    "workers": noises[label].workers,
                    "requests": noises[label].requests,
                    "duration_seconds": noises[label].duration,
                    "statuses": {str(status): count for status, count in noises[label].statuses.items()},
                    "errors": dict(noises[label].errors),
                },
                "requests": [asdict(result) for result in reports[label]],
            }
            for label in reports
        },
    }
    path.write_text(json.dumps(data, indent=2))


def _parse_targets(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for value in values:
        label, separator, url = value.partition("=")
        if not separator or not label or not url:
            parser.error(f"--target expects LABEL=URL, got {value!r}")
        targets[label] = url
    return targets


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark chat completions phase by phase against one or more targets.",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="LABEL=URL",
        help="base URL to benchmark, repeatable (e.g. api=https://api.libertai.io)",
    )
    parser.add_argument(
        "--token", default=os.environ.get("LIBERTAI_API_KEY", ""), help="Bearer token (or LIBERTAI_API_KEY)"
    )
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--requests", "-n", type=int, default=20, help="requests per target")
    parser.add_argument("--concurrency", "-x", type=int, default=4, help="parallel requests")
    parser.add_argument(
        "--unauth-concurrency",
        type=int,
        default=0,
        metavar="N",
        help="background clients sending unauthenticated requests during the measured phase (0 = off)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt sent with every request")
    parser.add_argument(
        "--model-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="keep each measured request talking for N seconds via follow-up turns on the same connection",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="LEVEL",
        help="send reasoning_effort explicitly (GLM: low or high); use the same value on every target",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0, help="connect/read/write timeout (seconds)")
    parser.add_argument("--no-stream", action="store_true", help="disable SSE streaming")
    parser.add_argument("--cold", action="store_true", help="open a fresh connection for every request")
    parser.add_argument(
        "--idle-gap", type=float, default=0.0, help="seconds to idle before each request (keepalive TTL)"
    )
    parser.add_argument("--keepalive", type=float, default=90.0, help="client-side keepalive expiry (seconds)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup requests per target (excluded from stats)")
    parser.add_argument(
        "--cacert",
        action="append",
        type=Path,
        metavar="PEM",
        help="additional CA certificate file (repeatable); certs/*.crt are always loaded",
    )
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification (last resort)")
    parser.add_argument("--json", type=Path, help="write raw per-request results to this file")
    args = parser.parse_args()

    if not args.token:
        parser.error("--token or LIBERTAI_API_KEY is required")
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    if args.unauth_concurrency < 0:
        parser.error("--unauth-concurrency cannot be negative")
    if args.model_wait > 0 and args.no_stream:
        parser.error("--model-wait requires streaming")
    if args.model_wait > 0:
        # Up to ~8k tokens per turn: one turn must not be so long that it hides
        # the turn boundary, and the client stops between turns once N elapsed.
        args.max_tokens = max(args.max_tokens, min(8192, math.ceil(args.model_wait * 100)))
    targets = _parse_targets(parser, args.target)

    reports: dict[str, list[RequestResult]] = {}
    walls: dict[str, float] = {}
    noises: dict[str, NoiseStats] = {}
    for label, base_url in targets.items():
        print(f"Benchmarking {label} ({base_url}) model={args.model} stream={not args.no_stream} ...", flush=True)
        reports[label], walls[label], noises[label] = await _run_target(args, base_url)
        _print_report(
            label,
            base_url,
            reports[label],
            walls[label],
            streaming=not args.no_stream,
            noise=noises[label],
            model_wait=args.model_wait,
        )

    _print_comparison(reports)
    if args.json:
        _write_json(args.json, args, targets, walls, reports, noises)
        print(f"\nRaw results written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
