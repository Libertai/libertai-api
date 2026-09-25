#!/usr/bin/env python3
"""Add or remove a temporary model redirection in the Aleph LTAI_PRICING aggregate.

The proxy resolves redirected model names at request time (aleph_service.resolve),
so a redirect published here takes effect on all replicas within one job cycle
(~30s) — useful when a model's server goes down for maintenance. Redirect the
base model name to cover both it and its -thinking variant (thinking is
preserved via is_reasoning_model).

Usage:
    python scripts/set_model_redirect.py --from qwen3.6-35b-a3b --to qwen3.8-27b
    python scripts/set_model_redirect.py --from qwen3.6-35b-a3b --remove

Requires ALEPH_SENDER_PRIVATE_KEY in the environment; the account must own the
LTAI_PRICING aggregate (ALEPH_AGGREGATE_ADDRESS, defaulting to the address
src/aleph.py reads).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import httpx
from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.client import AuthenticatedAlephHttpClient
from dotenv import load_dotenv

AGGREGATE_ADDRESS = "0xe1F7220D201C64871Cefb25320a8a588393eE508"
AGGREGATE_KEY = "LTAI_PRICING"
AGGREGATE_CHANNEL = "ALEPH-CLOUDSOLUTIONS"
DEFAULT_DESCRIPTION = "Temporary redirection"
VERIFY_ATTEMPTS = 5
VERIFY_INTERVAL = 3.0


async def fetch_aggregate(client: httpx.AsyncClient) -> dict:
    response = await client.get(
        f"https://api2.aleph.im/api/v0/aggregates/{AGGREGATE_ADDRESS}.json?keys={AGGREGATE_KEY}"
    )
    response.raise_for_status()
    return dict(response.json().get("data", {}).get(AGGREGATE_KEY, {}))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Set or remove a temporary model redirection")
    parser.add_argument("--from", dest="from_model", required=True, help="Model name to redirect")
    parser.add_argument("--to", dest="to_model", help="Target model (required unless --remove)")
    parser.add_argument("--remove", action="store_true", help="Remove the redirection instead of adding it")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--channel", default=AGGREGATE_CHANNEL)
    parser.add_argument(
        "--aggregate-address",
        default=os.environ.get("ALEPH_AGGREGATE_ADDRESS", AGGREGATE_ADDRESS),
    )
    args = parser.parse_args()

    load_dotenv()
    private_key = os.environ.get("ALEPH_SENDER_PRIVATE_KEY", "")
    if not private_key:
        raise SystemExit("ALEPH_SENDER_PRIVATE_KEY not set in the environment")

    if not args.remove and not args.to_model:
        raise SystemExit("--to is required unless --remove is set")

    account = ETHAccount(private_key=bytes.fromhex(private_key.removeprefix("0x")))
    sender_address = account.get_address()
    if sender_address.lower() != args.aggregate_address.lower():
        raise SystemExit(
            f"Sender {sender_address} does not own the aggregate {args.aggregate_address} — "
            f"the proxy would never read a redirect published under the sender's address"
        )

    async with httpx.AsyncClient(timeout=30.0) as http:
        content = await fetch_aggregate(http)

        redirections = [dict(r) for r in content.get("redirections", [])]
        from_model = args.from_model.lower()
        if args.remove:
            before = len(redirections)
            redirections = [r for r in redirections if r.get("from", "").lower() != from_model]
            print(f"Removing {before - len(redirections)} redirection(s) from '{from_model}'")
        else:
            to_model = args.to_model.lower()
            redirections = [r for r in redirections if r.get("from", "").lower() != from_model]
            redirections.append(
                {
                    "to": to_model,
                    "from": from_model,
                    "type": "MAINTENANCE",
                    "category": "text",
                    "description": args.description,
                }
            )
            print(f"Redirecting '{from_model}' -> '{to_model}'")
        content["redirections"] = redirections

        async with AuthenticatedAlephHttpClient(account=account) as client:
            message, _status = await client.create_aggregate(
                key=AGGREGATE_KEY,
                content=content,
                channel=args.channel,
            )
        print(f"Published aggregate update (message hash: {message.item_hash})")

        # Verify the update through the same endpoint the proxy reads; Aleph
        # propagation is eventually consistent, so retry briefly.
        for attempt in range(1, VERIFY_ATTEMPTS + 1):
            await asyncio.sleep(VERIFY_INTERVAL)
            current = await fetch_aggregate(http)
            live = {r.get("from", "").lower(): r.get("to", "") for r in current.get("redirections", [])}
            if args.remove and from_model not in live:
                print(f"Verified: no redirect from '{from_model}' remains")
                return
            if not args.remove and live.get(from_model) == args.to_model.lower():
                print(f"Verified: '{from_model}' -> '{live[from_model]}'")
                return
            print(f"Attempt {attempt}/{VERIFY_ATTEMPTS}: aggregate not updated yet (network propagation)")

    print(
        "Warning: could not verify the update through the aggregates endpoint — "
        f"check the message hash above and https://api.libertai.io/libertai/models "
        f"(published at {time.strftime('%H:%M:%S')})"
    )


if __name__ == "__main__":
    asyncio.run(main())
