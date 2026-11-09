#!/usr/bin/env python3
"""Update text-model prices in the Aleph LTAI_PRICING aggregate.

The aggregate is the billing source of truth for libertai-api,
libertai-inference, the website and the OpenRouter feed, so one publish updates
all of them on their next fetch (libertai-api within ~30s).

Usage:
    python scripts/set_model_prices.py --dry-run
    python scripts/set_model_prices.py
    python scripts/set_model_prices.py --table scripts/prices-2026-11.json

Requires ALEPH_SENDER_PRIVATE_KEY in the environment; the account must own the
LTAI_PRICING aggregate (ALEPH_AGGREGATE_ADDRESS, defaulting to the address
src/aleph.py reads).

The fetch-modify-publish window is last-write-wins on Aleph: a concurrent
aggregate update published between the fetch and the publish is clobbered
(fine for a single-operator tool). The verification loop only confirms this
script's own change, not that the rest of the content survived the window.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path

import httpx
from aleph.sdk.chains.ethereum import ETHAccount
from aleph.sdk.client import AuthenticatedAlephHttpClient
from dotenv import load_dotenv

AGGREGATE_ADDRESS = "0x238224C744F4b90b4494516e074D2676ECfC6803"
AGGREGATE_KEY = "LTAI_PRICING"
AGGREGATE_CHANNEL = "ALEPH-CLOUDSOLUTIONS"
VERIFY_ATTEMPTS = 5
VERIFY_INTERVAL = 3.0

# 2026-11 OpenRouter repricing: only the models priced below the market median.
# Prices are USD per 1M tokens. Keep qwen3.8-27b and its -tee deployment equal:
# the TEE endpoint carries no privacy premium.
DEFAULT_PRICES = {
    "qwen3.8-27b": {"input": 0.18, "output": 1.50},
    "qwen3.8-27b-tee": {"input": 0.18, "output": 1.50},
    "qwen3.6-35b-a3b": {"input": 0.15, "output": 0.90},
    "qwen3.5-122b-a10b": {"input": 0.28, "output": 2.10},
}


def apply_prices(models: list[dict], prices: dict[str, dict[str, float]]) -> list[str]:
    """Apply a price table to aggregate model entries in place.

    Returns the ids that changed. Raises if a target model is missing or has no
    text pricing, so a typo cannot publish a partial repricing.
    """
    by_id = {m.get("id", "").lower(): m for m in models}
    missing = [mid for mid in prices if mid.lower() not in by_id]
    if missing:
        raise ValueError(f"Models not found in the aggregate: {', '.join(sorted(missing))}")

    changed = []
    for model_id, price in prices.items():
        model = by_id[model_id.lower()]
        text = (model.get("pricing") or {}).get("text")
        if not isinstance(text, dict):
            raise TypeError(f"Model '{model_id}' has no text pricing")
        before = (text.get("price_per_million_input_tokens"), text.get("price_per_million_output_tokens"))
        text["price_per_million_input_tokens"] = price["input"]
        text["price_per_million_output_tokens"] = price["output"]
        if before != (price["input"], price["output"]):
            changed.append(model.get("id") or model_id)
    return changed


async def fetch_aggregate(client: httpx.AsyncClient, aggregate_address: str) -> dict:
    response = await client.get(
        f"https://api2.aleph.im/api/v0/aggregates/{aggregate_address}.json?keys={AGGREGATE_KEY}"
    )
    response.raise_for_status()
    return dict(response.json().get("data", {}).get(AGGREGATE_KEY, {}))


def load_prices(path: str | None) -> dict[str, dict[str, float]]:
    if path is None:
        return copy.deepcopy(DEFAULT_PRICES)
    table = json.loads(Path(path).read_text())
    if not isinstance(table, dict) or not table:
        raise SystemExit(f"{path}: expected a non-empty JSON object of model_id -> {{input, output}}")
    for model_id, price in table.items():
        if not isinstance(price, dict) or not {"input", "output"} <= set(price):
            raise SystemExit(f"{path}: '{model_id}' must have numeric 'input' and 'output'")
        if not all(isinstance(price[k], (int, float)) and price[k] >= 0 for k in ("input", "output")):
            raise SystemExit(f"{path}: '{model_id}' prices must be non-negative numbers")
    return table


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Update text-model prices in LTAI_PRICING")
    parser.add_argument("--table", help="JSON file of model_id -> {input, output}; defaults to the 2026-11 table")
    parser.add_argument("--dry-run", action="store_true", help="Print the diff without publishing")
    parser.add_argument("--channel", default=AGGREGATE_CHANNEL)
    parser.add_argument(
        "--aggregate-address",
        default=os.environ.get("ALEPH_AGGREGATE_ADDRESS") or AGGREGATE_ADDRESS,
    )
    args = parser.parse_args()

    prices = load_prices(args.table)

    if args.dry_run:
        async with httpx.AsyncClient(timeout=30.0) as http:
            content = await fetch_aggregate(http, args.aggregate_address)
        models = content.get("models") or []
        if not models:
            raise SystemExit("Aggregate returned no 'models' under LTAI_PRICING")
        before = {
            m.get("id", "").lower(): (
                m.get("pricing", {}).get("text", {}).get("price_per_million_input_tokens"),
                m.get("pricing", {}).get("text", {}).get("price_per_million_output_tokens"),
            )
            for m in models
        }
        apply_prices(models, prices)
        for model in models:
            model_id = model.get("id", "").lower()
            if model_id in {m.lower() for m in prices}:
                text = model["pricing"]["text"]
                old = before[model_id]
                print(
                    f"{model['id']}: "
                    f"${old[0]:.2f} in / ${old[1]:.2f} out -> "
                    f"${text['price_per_million_input_tokens']:.2f} in / "
                    f"${text['price_per_million_output_tokens']:.2f} out"
                )
        print("Dry run: nothing published")
        return

    private_key = os.environ.get("ALEPH_SENDER_PRIVATE_KEY", "")
    if not private_key:
        raise SystemExit("ALEPH_SENDER_PRIVATE_KEY not set in the environment")

    account = ETHAccount(private_key=bytes.fromhex(private_key.removeprefix("0x")))
    sender_address = account.get_address()
    if sender_address.lower() != args.aggregate_address.lower():
        raise SystemExit(
            f"Sender {sender_address} does not own the aggregate {args.aggregate_address} — "
            f"the API would never read prices published under the sender's address"
        )

    async with httpx.AsyncClient(timeout=30.0) as http:
        content = await fetch_aggregate(http, args.aggregate_address)
        models = content.get("models") or []
        if not models:
            raise SystemExit(
                f"Aggregate {args.aggregate_address} returned no 'models' under key "
                f"'{AGGREGATE_KEY}' — refusing to publish over it (the API would lose its model list)"
            )

        changed = apply_prices(models, prices)
        if not changed:
            print("Prices already match the table; nothing to publish")
            return
        print(f"Updating {len(changed)} model(s): {', '.join(changed)}")

        async with AuthenticatedAlephHttpClient(account=account) as client:
            # Explicit address: the SDK otherwise falls back to the ADDRESS_TO_USE
            # env var, which could publish away from the ownership-checked address.
            message, _status = await client.create_aggregate(
                key=AGGREGATE_KEY,
                content=content,
                address=args.aggregate_address,
                channel=args.channel,
            )
        print(f"Published aggregate update: {message.item_hash}")

        # Verify through the same endpoint the API reads; Aleph propagation is
        # eventually consistent, so retry briefly.
        expected = {m.lower(): (p["input"], p["output"]) for m, p in prices.items()}
        for attempt in range(1, VERIFY_ATTEMPTS + 1):
            await asyncio.sleep(VERIFY_INTERVAL)
            current = await fetch_aggregate(http, args.aggregate_address)
            live = {
                m.get("id", "").lower(): (
                    m.get("pricing", {}).get("text", {}).get("price_per_million_input_tokens"),
                    m.get("pricing", {}).get("text", {}).get("price_per_million_output_tokens"),
                )
                for m in current.get("models", [])
            }
            if all(live.get(mid) == want for mid, want in expected.items()):
                print("Verified: live aggregate matches the new prices")
                return
            print(f"Attempt {attempt}/{VERIFY_ATTEMPTS}: aggregate not updated yet (network propagation)")

    print(
        "Warning: could not verify the update through the aggregates endpoint — "
        f"check the message hash above and https://api.libertai.io/v1/models "
        f"(published at {time.strftime('%H:%M:%S')})"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
