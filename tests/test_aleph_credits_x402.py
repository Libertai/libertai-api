import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from x402_helpers import REQUIREMENTS, payment_header

from src import aleph_credits
from src.x402 import FacilitatorUnavailable

# REQUIREMENTS quotes 5000 atomic USDC = $0.005.
AMOUNT = 0.005


@pytest.fixture
def credits(monkeypatch):
    """Credits endpoint with x402 and Aleph stubbed; returns (client, events) where events logs settle/transfer order."""
    events: list[str] = []
    used: set[str] = set()

    async def _requirements(scheme, price):
        return REQUIREMENTS

    async def _verify(payment):
        return None

    async def _reserve(payment):
        key = json.dumps(payment["payload"], sort_keys=True)
        if key in used:
            return False
        used.add(key)
        return True

    async def _settle(payment):
        events.append("settle")
        return True

    class _AlephClient:
        def __init__(self, account):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def create_post(self, **kwargs):
            events.append("transfer")
            return SimpleNamespace(item_hash="hash"), None

    monkeypatch.setattr(aleph_credits, "_aleph_account", object())
    monkeypatch.setattr(aleph_credits, "AuthenticatedAlephHttpClient", _AlephClient)
    monkeypatch.setattr(aleph_credits.config, "X402_MIN_PRICE", 0.005)
    monkeypatch.setattr(aleph_credits.x402_manager, "build_requirements", _requirements)
    monkeypatch.setattr(aleph_credits.x402_manager, "verify_payment", _verify)
    monkeypatch.setattr(aleph_credits.x402_manager, "settle_payment", _settle)

    async def _release(payment):
        events.append("release")
        used.discard(json.dumps(payment["payload"], sort_keys=True))

    monkeypatch.setattr(aleph_credits, "reserve_payment", _reserve)
    monkeypatch.setattr(aleph_credits, "release_payment", _release)

    app = FastAPI()
    app.include_router(aleph_credits.router)
    return TestClient(app), events


def _buy(client, amount=AMOUNT, header=None):
    headers = {"payment-signature": header} if header else {}
    return client.post("/libertai/aleph-credits", json={"address": "0xabc", "amount": amount}, headers=headers)


def test_amount_below_minimum_is_rejected(credits):
    client, events = credits
    assert _buy(client, amount=0.001).status_code == 400
    assert not events


def test_without_payment_answers_402(credits):
    client, events = credits
    assert _buy(client).status_code == 402
    assert not events


def test_overpaying_payment_is_rejected(credits):
    client, events = credits
    resp = _buy(client, header=payment_header({**REQUIREMENTS, "amount": "6000"}))
    assert resp.status_code == 402
    assert not events


def test_paid_purchase_settles_before_transferring(credits):
    client, events = credits
    resp = _buy(client, header=payment_header(REQUIREMENTS))
    assert resp.status_code == 200
    assert resp.json()["credits_transferred"] == 5000
    assert events == ["settle", "transfer"]


def test_reused_payment_is_rejected(credits):
    client, events = credits
    header = payment_header(REQUIREMENTS)
    assert _buy(client, header=header).status_code == 200
    resp = _buy(client, header=header)
    assert resp.status_code == 402
    assert resp.json()["error"] == "Payment already used"
    assert events == ["settle", "transfer"]


def test_facilitator_down_answers_503(credits, monkeypatch):
    async def _down(payment):
        raise FacilitatorUnavailable("down")

    monkeypatch.setattr(aleph_credits.x402_manager, "verify_payment", _down)
    client, events = credits
    assert _buy(client, header=payment_header(REQUIREMENTS)).status_code == 503
    assert not events


def test_failed_settlement_transfers_nothing(credits, monkeypatch):
    async def _fail(payment):
        return False

    monkeypatch.setattr(aleph_credits.x402_manager, "settle_payment", _fail)
    client, events = credits
    header = payment_header(REQUIREMENTS)
    assert _buy(client, header=header).status_code == 500
    assert events == ["release"]
    # Released, so the same signature can be retried.
    monkeypatch.setattr(aleph_credits.x402_manager, "settle_payment", lambda payment: _ok())
    assert _buy(client, header=header).status_code == 200


async def _ok():
    return True


def test_credits_equal_the_charged_amount(credits, monkeypatch):
    # 2.01 * 1e6 floors to 2009999 in float math; the charge is 2010000.
    charged = {**REQUIREMENTS, "amount": "2010000"}

    async def _requirements(scheme, price):
        return charged

    monkeypatch.setattr(aleph_credits.x402_manager, "build_requirements", _requirements)
    client, _ = credits
    resp = _buy(client, amount=2.01, header=payment_header(charged))
    assert resp.json()["credits_transferred"] == 2010000
