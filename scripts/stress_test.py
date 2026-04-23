#!/usr/bin/env python3
"""Stress-test the LibertAI API with real inference requests, rotating models.

Reports ongoing progress and final stats (success rate, latency percentiles,
per-model success, per-status-code distribution).

Usage:
    python scripts/stress_test.py \
        --url https://api.libertai.io \
        --key $LIBERTAI_API_KEY \
        --concurrency 10 \
        --duration 60

During a rolling deploy this gives you a continuous signal on whether any
requests are being dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
import signal
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import httpx


PROMPTS = [
    "Say hi in one word.",
    "Reply with a single number between 1 and 10.",
    "What is 2+2? Reply with just the number.",
    "Name one color. One word only.",
    "Reply with exactly 'ok'.",
]


@dataclass
class Result:
    ok: bool
    status: int
    latency_ms: float
    model: str
    error: str | None = None


@dataclass
class Stats:
    results: list[Result] = field(default_factory=list)
    start: float = field(default_factory=time.monotonic)

    def add(self, r: Result) -> None:
        self.results.append(r)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def summary(self) -> str:
        if not self.results:
            return "no requests completed"
        total = self.total
        success = self.success
        rate = success / total * 100
        rps = total / self.elapsed if self.elapsed > 0 else 0

        latencies = sorted(r.latency_ms for r in self.results if r.ok)
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
        avg = statistics.mean(latencies) if latencies else 0

        by_status: Counter[int] = Counter(r.status for r in self.results)
        by_model_total: Counter[str] = Counter(r.model for r in self.results)
        by_model_ok: Counter[str] = Counter(r.model for r in self.results if r.ok)

        errors = [r for r in self.results if r.error]
        error_samples: Counter[str] = Counter(r.error for r in errors if r.error)

        lines = [
            f"\n{'=' * 60}",
            f"Duration:      {self.elapsed:.1f}s",
            f"Total:         {total}",
            f"Success:       {success}  ({rate:.2f}%)",
            f"Failures:      {total - success}",
            f"Throughput:    {rps:.1f} req/s",
            f"Latency (ok):  avg={avg:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms",
            "",
            "By status code:",
        ]
        for status, n in sorted(by_status.items()):
            lines.append(f"  {status}: {n}")

        lines.append("")
        lines.append("By model (success/total):")
        for model in sorted(by_model_total):
            lines.append(f"  {model}: {by_model_ok[model]}/{by_model_total[model]}")

        if error_samples:
            lines.append("")
            lines.append("Error samples:")
            for err, n in error_samples.most_common(5):
                lines.append(f"  [{n}x] {err[:140]}")

        lines.append("=" * 60)
        return "\n".join(lines)


IMAGE_MODEL_HINTS = ("z-image", "flux", "sdxl", "stable-diffusion", "dall-e", "image-")


async def fetch_models(client: httpx.AsyncClient, url: str, exclude: list[str]) -> list[str]:
    r = await client.get(f"{url}/v1/models", timeout=15.0)
    r.raise_for_status()
    data = r.json().get("data", [])
    models = []
    for m in data:
        mid = m["id"]
        if mid.endswith("-thinking"):
            continue
        if mid in exclude:
            continue
        if any(hint in mid.lower() for hint in IMAGE_MODEL_HINTS):
            continue
        models.append(mid)
    if not models:
        raise RuntimeError(f"No chat models returned from {url}/v1/models")
    return models


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    model: str,
    prompt: str,
    timeout: float,
) -> Result:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "stream": False,
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions", json=body, headers=headers, timeout=timeout
        )
        latency = (time.monotonic() - t0) * 1000
        ok = 200 <= r.status_code < 300
        err = None if ok else r.text[:200]
        return Result(ok=ok, status=r.status_code, latency_ms=latency, model=model, error=err)
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return Result(
            ok=False, status=0, latency_ms=latency, model=model, error=f"{type(e).__name__}: {e}"
        )


async def worker(
    worker_id: int,
    client: httpx.AsyncClient,
    url: str,
    key: str,
    models: list[str],
    stop: asyncio.Event,
    stats: Stats,
    timeout: float,
) -> None:
    # Each worker gets its own rotation offset so they don't hammer the same model.
    offset = worker_id
    for i in itertools.count():
        if stop.is_set():
            return
        model = models[(offset + i) % len(models)]
        prompt = random.choice(PROMPTS)
        r = await one_request(client, url, key, model, prompt, timeout)
        stats.add(r)


async def ticker(stats: Stats, stop: asyncio.Event, interval: float) -> None:
    last_total = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        total = stats.total
        success = stats.success
        recent = total - last_total
        last_total = total
        rate = (success / total * 100) if total else 0
        print(
            f"[{stats.elapsed:5.1f}s] total={total}  success={success} ({rate:5.1f}%)  "
            f"recent={recent}/{interval:.0f}s",
            flush=True,
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="https://api.libertai.io", help="API base URL")
    ap.add_argument("--key", required=True, help="API key (Bearer token)")
    ap.add_argument("--concurrency", type=int, default=5, help="Concurrent workers")
    ap.add_argument("--duration", type=int, default=60, help="Run time in seconds")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    ap.add_argument("--report-every", type=float, default=5.0, help="Progress report interval")
    ap.add_argument(
        "--models",
        nargs="+",
        help="Explicit model list (overrides /v1/models)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Model IDs to skip (in addition to auto-detected image models)",
    )
    args = ap.parse_args()

    stats = Stats()
    stop = asyncio.Event()

    # Ctrl-C: print summary before exiting.
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    limits = httpx.Limits(
        max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(limits=limits) as client:
        if args.models:
            models = args.models
        else:
            print(f"Fetching model list from {args.url}/v1/models ...", flush=True)
            models = await fetch_models(client, args.url, args.exclude)
        print(f"Using {len(models)} models: {', '.join(models[:10])}{'...' if len(models) > 10 else ''}")
        print(
            f"Stress-testing {args.url} for {args.duration}s with concurrency={args.concurrency}"
        )
        print()

        async def stopper():
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                stop.set()

        workers = [
            asyncio.create_task(
                worker(i, client, args.url, args.key, models, stop, stats, args.timeout)
            )
            for i in range(args.concurrency)
        ]
        tick_task = asyncio.create_task(ticker(stats, stop, args.report_every))
        stop_task = asyncio.create_task(stopper())

        try:
            await stop_task
        finally:
            stop.set()
            await asyncio.gather(*workers, return_exceptions=True)
            tick_task.cancel()
            await asyncio.gather(tick_task, return_exceptions=True)

    print(stats.summary())


if __name__ == "__main__":
    asyncio.run(main())
