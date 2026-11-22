import asyncio
import base64
import json

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from x402_helpers import REQUIREMENTS, payment_header

from src import x402
from src.cdp_auth import cdp_auth_headers
from src.x402 import (
    X402Manager,
    accepted_matches,
    decode_payment_header,
    outlives_settlement,
    payment_valid_until,
    usd_to_atomic,
)

FACILITATOR = "0x00000000000000000000000000000000000000Fa"


@pytest.fixture
def manager(monkeypatch):
    mgr = X402Manager()
    mgr.prices = {
        "text": {"price_per_million_input_tokens": 1.0, "price_per_million_output_tokens": 2.0},
        "img": {"price_per_image": 0.04},
        "emb": {"is_embedding": True, "price_per_million_input_tokens": 0.01},
    }

    async def _kind_extra(self):
        return {"facilitatorAddress": FACILITATOR}

    monkeypatch.setattr(X402Manager, "get_upto_kind_extra", _kind_extra)
    monkeypatch.setattr(x402.config, "X402_MIN_PRICE", 0.005)
    monkeypatch.setattr(x402.config, "X402_WALLET_ADDRESS", REQUIREMENTS["payTo"])
    return mgr


def test_text_is_upto_with_max_tokens_in_price(manager):
    price, scheme = asyncio.run(manager.compute_price("text", {"messages": [], "max_tokens": 1_000_000}))
    assert scheme == "upto"
    assert price == pytest.approx(2.0, rel=1e-3)


def test_fixed_price_models_are_exact(manager):
    assert asyncio.run(manager.compute_price("img", {})) == (0.04, "exact")
    assert asyncio.run(manager.compute_price("emb", {"input": "hi"})) == (0.005, "exact")


def test_unknown_model_has_no_price(manager):
    assert asyncio.run(manager.compute_price("nope", {})) is None


def test_usd_to_atomic_rounds_up():
    assert usd_to_atomic(0.005) == "5000"
    assert usd_to_atomic(0.0000001) == "1"
    assert usd_to_atomic(0.0050001) == "5001"


def test_upto_requirements_use_permit2_and_facilitator(manager):
    req = asyncio.run(manager.build_requirements("upto", 0.01))
    assert req["scheme"] == "upto"
    assert req["amount"] == "10000"
    assert req["network"] == "eip155:8453"
    assert req["payTo"] == REQUIREMENTS["payTo"]
    assert req["extra"] == {
        "name": "USD Coin",
        "version": "2",
        "assetTransferMethod": "permit2",
        "facilitatorAddress": FACILITATOR,
    }


def test_exact_requirements_have_no_upto_fields(manager):
    req = asyncio.run(manager.build_requirements("exact", 0.04))
    assert req["extra"] == {"name": "USD Coin", "version": "2"}


def test_upto_requirements_unavailable_without_facilitator(manager, monkeypatch):
    async def _none(self):
        return None

    monkeypatch.setattr(X402Manager, "get_upto_kind_extra", _none)
    assert asyncio.run(manager.build_requirements("upto", 0.01)) is None


def test_402_carries_v2_payment_required_in_header_and_body():
    resp = X402Manager.build_402_response(REQUIREMENTS, "https://api/x", "desc")
    assert resp.status_code == 402
    from_header = json.loads(base64.b64decode(resp.headers["PAYMENT-REQUIRED"]))
    assert from_header == json.loads(resp.body)
    assert from_header["x402Version"] == 2
    assert from_header["resource"] == {"url": "https://api/x", "description": "desc", "mimeType": "application/json"}
    assert from_header["accepts"] == [REQUIREMENTS]


@pytest.mark.parametrize(
    "header",
    [
        "not base64!",
        base64.b64encode(b"not json").decode(),
        payment_header(REQUIREMENTS, x402Version=1),
        payment_header(REQUIREMENTS, payload="x"),
        base64.b64encode(json.dumps({"x402Version": 2, "payload": {}}).encode()).decode(),
    ],
)
def test_malformed_payment_headers_are_rejected(header):
    assert decode_payment_header(header) is None


def test_valid_payment_header_decodes():
    assert decode_payment_header(payment_header(REQUIREMENTS))["accepted"] == REQUIREMENTS


def test_accepted_may_overpay_and_differ_in_address_case():
    accepted = {**REQUIREMENTS, "amount": "6000", "payTo": REQUIREMENTS["payTo"].lower()}
    assert accepted_matches(accepted, REQUIREMENTS)


@pytest.mark.parametrize(
    "change",
    [
        {"amount": "4999"},
        {"amount": "abc"},
        {"scheme": "upto"},
        {"network": "eip155:84532"},
        {"payTo": "0x9999999999999999999999999999999999999999"},
        {"asset": "0x0000000000000000000000000000000000000001"},
        {"extra": {"name": "USDC", "version": "2"}},
    ],
)
def test_accepted_mismatch_is_rejected(change):
    assert not accepted_matches({**REQUIREMENTS, **change}, REQUIREMENTS)


def test_accepted_must_name_our_upto_facilitator():
    required = {
        **REQUIREMENTS,
        "scheme": "upto",
        "extra": {**REQUIREMENTS["extra"], "facilitatorAddress": FACILITATOR},
    }
    assert accepted_matches(
        {**required, "extra": {**required["extra"], "facilitatorAddress": FACILITATOR.lower()}}, required
    )
    other = {**required, "extra": {**required["extra"], "facilitatorAddress": "0x" + "1" * 40}}
    assert not accepted_matches(other, required)


def _decode(headers, public_key, algorithm):
    token = headers["Authorization"].removeprefix("Bearer ")
    return jwt.decode(token, public_key, algorithms=[algorithm], options={"verify_aud": False})


def test_cdp_jwt_with_ec_pem_key():
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    ).decode()
    # As stored in an env var with escaped newlines.
    headers = cdp_auth_headers("kid", pem.replace("\n", "\\n"), "POST", "/platform/v2/x402/verify")
    claims = _decode(headers, key.public_key(), "ES256")
    assert claims["sub"] == "kid"
    assert claims["iss"] == "cdp"
    assert claims["uris"] == ["POST api.cdp.coinbase.com/platform/v2/x402/verify"]


def test_cdp_jwt_with_ed25519_key():
    key = ed25519.Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    headers = cdp_auth_headers("kid", base64.b64encode(raw + pub).decode(), "GET", "/platform/v2/x402/supported")
    claims = _decode(headers, key.public_key(), "EdDSA")
    assert claims["uris"] == ["GET api.cdp.coinbase.com/platform/v2/x402/supported"]
    assert jwt.get_unverified_header(headers["Authorization"].removeprefix("Bearer "))["kid"] == "kid"


@pytest.fixture
def paid_proxy(monkeypatch):
    """Proxy wired to one upstream with x402 pricing stubbed; returns (client, sent requests, verify calls)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src import proxy

    monkeypatch.setattr(proxy.config, "MODELS", {"m": ["http://up"]})
    monkeypatch.setattr(proxy.aleph_service, "resolve", lambda model: model)

    async def _loads(_model):
        return {"http://up": 0}

    async def _noop(*args, **kwargs):
        return None

    async def _price(model, body):
        return 0.005, "exact"

    async def _requirements(scheme, price):
        return REQUIREMENTS

    verified: list[dict] = []

    async def _verify(payment):
        verified.append(payment)

    sent: list = []

    async def _send(req, **kwargs):
        import httpx

        async def _chunks():
            yield b'{"ok": true}'

        sent.append(req)
        resp = httpx.Response(200, content=_chunks(), headers={"Content-Type": "application/json"})
        resp.request = req
        return resp

    monkeypatch.setattr(proxy, "get_model_loads", _loads)
    monkeypatch.setattr(proxy, "load_acquire", _noop)
    monkeypatch.setattr(proxy, "load_release", _noop)
    monkeypatch.setattr(proxy.x402_manager, "compute_price", _price)
    monkeypatch.setattr(proxy.x402_manager, "build_requirements", _requirements)
    monkeypatch.setattr(proxy.x402_manager, "verify_payment", _verify)
    used: set[str] = set()

    async def _reserve(payment):
        key = json.dumps(payment["payload"], sort_keys=True)
        if key in used:
            return False
        used.add(key)
        return True

    monkeypatch.setattr(proxy, "reserve_payment", _reserve)
    monkeypatch.setattr(proxy.client, "send", _send)

    app = FastAPI()
    app.include_router(proxy.router)
    return TestClient(app), sent, verified


def test_proxy_without_payment_answers_402(paid_proxy):
    client, sent, verified = paid_proxy
    resp = client.post("/v1/chat/completions", json={"model": "m"})
    assert resp.status_code == 402
    assert json.loads(base64.b64decode(resp.headers["payment-required"]))["accepts"] == [REQUIREMENTS]
    assert not sent and not verified


def test_proxy_rejects_payment_for_other_requirements_without_verifying(paid_proxy):
    client, sent, verified = paid_proxy
    header = payment_header({**REQUIREMENTS, "amount": "1"})
    resp = client.post("/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": header})
    assert resp.status_code == 402
    assert not sent and not verified


def test_proxy_ignores_legacy_x_payment_header(paid_proxy):
    client, sent, _ = paid_proxy
    resp = client.post(
        "/v1/chat/completions", json={"model": "m"}, headers={"x-payment": payment_header(REQUIREMENTS)}
    )
    assert resp.status_code == 402
    assert not sent


def test_proxy_forwards_verified_payment_for_settlement(paid_proxy):
    client, sent, verified = paid_proxy
    header = payment_header(REQUIREMENTS)
    resp = client.post("/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": header})
    assert resp.status_code == 200
    assert len(verified) == 1
    (req,) = sent
    assert req.headers["x-payment"] == header
    assert json.loads(req.headers["x-payment-requirements"]) == REQUIREMENTS


def test_proxy_surfaces_facilitator_rejection_reason(paid_proxy, monkeypatch):
    from src import proxy

    async def _reject(payment):
        return "insufficient_funds"

    monkeypatch.setattr(proxy.x402_manager, "verify_payment", _reject)
    client, sent, _ = paid_proxy
    resp = client.post(
        "/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": payment_header(REQUIREMENTS)}
    )
    assert resp.status_code == 402
    assert resp.json()["error"] == "insufficient_funds"
    assert not sent


def test_text_price_uses_the_larger_token_limit(manager):
    body = {"messages": [], "max_tokens": 1, "max_completion_tokens": 1_000_000}
    price, _ = asyncio.run(manager.compute_price("text", body))
    assert price == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("bad", [-5, "1000000", 1e6, True])
def test_any_non_int_token_limit_prices_the_model_maximum(manager, bad):
    manager.prices["text"]["default_max_tokens"] = 1_000_000
    body = {"messages": [], "max_tokens": 1, "max_completion_tokens": bad}
    price, _ = asyncio.run(manager.compute_price("text", body))
    assert price == pytest.approx(2.0, rel=1e-3)


def test_validity_read_from_eip3009_and_permit2_payloads():
    upto = {**REQUIREMENTS, "scheme": "upto"}
    assert payment_valid_until({"accepted": REQUIREMENTS, "payload": {"authorization": {"validBefore": "123"}}}) == 123
    assert payment_valid_until({"accepted": upto, "payload": {"permit2Authorization": {"deadline": "456"}}}) == 456
    assert payment_valid_until({"accepted": REQUIREMENTS, "payload": {}}) is None


def test_short_lived_authorization_does_not_outlive_settlement():
    import time

    upto = {**REQUIREMENTS, "scheme": "upto"}
    soon = {"accepted": REQUIREMENTS, "payload": {"authorization": {"validBefore": str(int(time.time()) + 60)}}}
    later = {"accepted": upto, "payload": {"permit2Authorization": {"deadline": str(int(time.time()) + 3600)}}}
    assert not outlives_settlement(soon)
    assert outlives_settlement(later)


def test_proxy_rejects_replayed_payment(paid_proxy):
    client, sent, _ = paid_proxy
    header = payment_header(REQUIREMENTS)
    first = client.post("/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": header})
    second = client.post("/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": header})
    assert first.status_code == 200
    assert second.status_code == 402
    assert second.json()["error"] == "Payment already used"
    assert len(sent) == 1


def test_proxy_rejects_authorization_expiring_before_settlement(paid_proxy):
    import time

    client, sent, verified = paid_proxy
    header = payment_header(
        REQUIREMENTS,
        payload={"signature": "0x", "authorization": {"validBefore": str(int(time.time()) + 30)}},
    )
    resp = client.post("/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": header})
    assert resp.status_code == 402
    assert not sent and not verified


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}
        self.ttls: list[int] = []

    async def set(self, key, value, nx=False, ex=None):
        self.ttls.append(ex)
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


def _signed(payer="0xAbC", nonce="0x01", extra_payload=None, expiry_in=3600):
    import time

    authorization = {"from": payer, "nonce": nonce, "validBefore": str(int(time.time()) + expiry_in)}
    return {
        "accepted": REQUIREMENTS,
        "payload": {"signature": "0x", "authorization": authorization, **(extra_payload or {})},
    }


def test_reservation_is_keyed_on_signed_payer_and_nonce(monkeypatch):
    from src.x402 import reserve_payment

    redis = _FakeRedis()
    monkeypatch.setattr(x402, "get_redis", lambda: redis)
    assert asyncio.run(reserve_payment(_signed()))
    # Unsigned payload fields and payer address case don't make it a new payment.
    assert not asyncio.run(reserve_payment(_signed(payer="0xabc", extra_payload={"junk": 1})))
    assert asyncio.run(reserve_payment(_signed(nonce="0x02")))
    assert redis.ttls[0] >= 3600


def test_reservation_refuses_payment_without_payer_or_nonce(monkeypatch):
    from src.x402 import reserve_payment

    monkeypatch.setattr(x402, "get_redis", lambda: _FakeRedis())
    assert not asyncio.run(reserve_payment(_signed(payer=None)))
    assert not asyncio.run(reserve_payment(_signed(nonce=None)))
    assert not asyncio.run(reserve_payment({"accepted": REQUIREMENTS, "payload": {}}))


def test_shorter_max_timeout_is_rejected():
    assert not accepted_matches({**REQUIREMENTS, "maxTimeoutSeconds": 1}, REQUIREMENTS)
    assert accepted_matches({**REQUIREMENTS, "maxTimeoutSeconds": 7200}, REQUIREMENTS)


def _supported_response(kinds, signers=None):
    import httpx

    body = {"kinds": kinds, "signers": signers or {}}
    return httpx.Response(200, json=body, request=httpx.Request("GET", "https://x"))


@pytest.fixture
def fresh_supported(monkeypatch):
    monkeypatch.setattr(X402Manager, "_upto_kind_extra", None)
    monkeypatch.setattr(X402Manager, "_upto_kind_fetched_at", 0.0)
    monkeypatch.setattr(x402, "cdp_auth_headers", lambda *args: {})


def test_upto_facilitator_read_from_matching_supported_kind(monkeypatch, fresh_supported):
    kinds = [
        {"x402Version": 2, "scheme": "exact", "network": "eip155:8453", "extra": {"facilitatorAddress": "0xexact"}},
        {"x402Version": 2, "scheme": "upto", "network": "eip155:84532", "extra": {"facilitatorAddress": "0xtestnet"}},
        {"x402Version": 1, "scheme": "upto", "network": "eip155:8453", "extra": {"facilitatorAddress": "0xv1"}},
        {
            "x402Version": 2,
            "scheme": "upto",
            "network": "eip155:8453",
            "extra": {"facilitatorAddress": "0xright", "newField": 1},
        },
    ]
    calls = []

    async def _get(url, headers=None):
        calls.append(url)
        return _supported_response(kinds)

    monkeypatch.setattr(x402.client, "get", _get)
    mgr = X402Manager()
    expected = {"facilitatorAddress": "0xright", "newField": 1}
    assert asyncio.run(mgr.get_upto_kind_extra()) == expected
    assert asyncio.run(mgr.get_upto_kind_extra()) == expected
    assert calls == ["https://api.cdp.coinbase.com/platform/v2/x402/supported"]


def test_upto_facilitator_falls_back_to_stale_value_when_cdp_fails(monkeypatch, fresh_supported):
    async def _fail(url, headers=None):
        raise RuntimeError("down")

    monkeypatch.setattr(X402Manager, "_upto_kind_extra", {"facilitatorAddress": "0xstale"})
    monkeypatch.setattr(x402.client, "get", _fail)
    assert asyncio.run(X402Manager().get_upto_kind_extra()) == {"facilitatorAddress": "0xstale"}


def test_upto_facilitator_unknown_without_matching_kind(monkeypatch, fresh_supported):
    async def _get(url, headers=None):
        return _supported_response([{"x402Version": 2, "scheme": "exact", "network": "eip155:8453"}])

    monkeypatch.setattr(x402.client, "get", _get)
    assert asyncio.run(X402Manager().get_upto_kind_extra()) is None


@pytest.mark.parametrize("failure", ["raise", 502])
def test_verify_raises_when_facilitator_fails(monkeypatch, failure):
    import httpx

    from src.x402 import FacilitatorUnavailable

    async def _post(cls, endpoint, body, timeout=30.0):
        if failure == "raise":
            raise httpx.ConnectError("down")
        return httpx.Response(failure, json={"error": "bad gateway"})

    monkeypatch.setattr(X402Manager, "_facilitator_post", classmethod(_post))
    with pytest.raises(FacilitatorUnavailable):
        asyncio.run(X402Manager.verify_payment({"accepted": REQUIREMENTS, "payload": {}}))


@pytest.mark.parametrize("status", [200, 400])
def test_verify_returns_reason_for_rejected_payment(monkeypatch, status):
    import httpx

    async def _post(cls, endpoint, body, timeout=30.0):
        return httpx.Response(status, json={"isValid": False, "invalidReason": "insufficient_funds"})

    monkeypatch.setattr(X402Manager, "_facilitator_post", classmethod(_post))
    assert asyncio.run(X402Manager.verify_payment({"accepted": REQUIREMENTS, "payload": {}})) == "insufficient_funds"


def test_proxy_answers_503_when_facilitator_is_down(paid_proxy, monkeypatch):
    from src import proxy
    from src.x402 import FacilitatorUnavailable

    async def _down(payment):
        raise FacilitatorUnavailable("down")

    monkeypatch.setattr(proxy.x402_manager, "verify_payment", _down)
    client, sent, _ = paid_proxy
    resp = client.post(
        "/v1/chat/completions", json={"model": "m"}, headers={"payment-signature": payment_header(REQUIREMENTS)}
    )
    assert resp.status_code == 503
    assert not sent


def test_cors_exposes_payment_required_header():
    from fastapi.testclient import TestClient

    from src.server import app

    resp = TestClient(app).post(
        "/v1/chat/completions", json={"model": "nope"}, headers={"Origin": "https://example.com"}
    )
    assert "payment-required" in resp.headers.get("access-control-expose-headers", "").lower()


@pytest.mark.parametrize(("field", "value"), [("n", 3), ("best_of", "3"), ("n", 3.0)])
def test_text_price_covers_every_choice(manager, field, value):
    body = {"messages": [], "max_tokens": 100_000, field: value}
    price, _ = asyncio.run(manager.compute_price("text", body))
    assert price == pytest.approx(0.6, rel=1e-2)


@pytest.mark.parametrize("value", [None, 0, -2, "x", True, 2.5])
def test_unusable_choice_count_prices_one_choice(manager, value):
    body = {"messages": [], "max_tokens": 100_000, "n": value}
    price, _ = asyncio.run(manager.compute_price("text", body))
    assert price == pytest.approx(0.2, rel=1e-2)


def test_completions_prompt_is_priced(manager):
    short, _ = asyncio.run(manager.compute_price("text", {"prompt": "", "max_tokens": 1}))
    long, _ = asyncio.run(manager.compute_price("text", {"prompt": "word " * 100_000, "max_tokens": 1}))
    assert long > short


def test_authorization_read_from_the_field_of_the_accepted_scheme():
    upto = {**REQUIREMENTS, "scheme": "upto"}
    assert payment_valid_until({"accepted": upto, "payload": {"permit2Authorization": {"deadline": "5"}}}) == 5
    assert (
        payment_valid_until({"accepted": REQUIREMENTS, "payload": {"permit2Authorization": {"deadline": "5"}}}) is None
    )
    assert payment_valid_until({"accepted": upto, "payload": {"authorization": {"validBefore": "5"}}}) is None


def test_decoy_authorization_alongside_the_real_one_is_rejected(monkeypatch):
    from src.x402 import reserve_payment

    monkeypatch.setattr(x402, "get_redis", lambda: _FakeRedis())
    upto = {**REQUIREMENTS, "scheme": "upto"}
    payment = {
        "accepted": upto,
        "payload": {
            "permit2Authorization": {"from": "0xa", "nonce": "1", "deadline": "9999999999"},
            "authorization": {"from": "0xdecoy", "nonce": "2", "validBefore": "9999999999"},
        },
    }
    assert payment_valid_until(payment) is None
    assert not asyncio.run(reserve_payment(payment))


def test_reservation_ignores_nonce_spelling(monkeypatch):
    from src.x402 import reserve_payment

    redis = _FakeRedis()
    monkeypatch.setattr(x402, "get_redis", lambda: redis)
    assert asyncio.run(reserve_payment(_signed(nonce="0xAbC")))
    assert not asyncio.run(reserve_payment(_signed(nonce="0xabc")))
    assert not asyncio.run(reserve_payment(_signed(nonce="0x0000abc")))
    assert not asyncio.run(reserve_payment(_signed(nonce=str(0xABC))))
    assert not asyncio.run(reserve_payment(_signed(nonce="not-a-number")))


def test_exact_amount_rejects_overpayment():
    assert accepted_matches({**REQUIREMENTS, "amount": "6000"}, REQUIREMENTS)
    assert not accepted_matches({**REQUIREMENTS, "amount": "6000"}, REQUIREMENTS, exact_amount=True)
    assert accepted_matches(REQUIREMENTS, REQUIREMENTS, exact_amount=True)


@pytest.mark.parametrize(("status", "body"), [(401, {"error": "unauthorized"}), (404, {}), (429, {}), (200, ["x"])])
def test_verify_treats_non_judgement_responses_as_unavailable(monkeypatch, status, body):
    import httpx

    from src.x402 import FacilitatorUnavailable

    async def _post(cls, endpoint, payload, timeout=30.0):
        return httpx.Response(status, json=body)

    monkeypatch.setattr(X402Manager, "_facilitator_post", classmethod(_post))
    with pytest.raises(FacilitatorUnavailable):
        asyncio.run(X402Manager.verify_payment({"accepted": REQUIREMENTS, "payload": {}}))


def test_image_price_covers_every_image(manager):
    assert asyncio.run(manager.compute_price("img", {"n": 3})) == (pytest.approx(0.12), "exact")
    assert asyncio.run(manager.compute_price("img", {"n": "x"})) == (0.04, "exact")


def test_text_input_counts_every_body_field(manager):
    base = {"messages": [], "max_tokens": 1}
    plain, _ = asyncio.run(manager.compute_price("text", base))
    with_tools, _ = asyncio.run(manager.compute_price("text", {**base, "tools": [{"description": "word " * 100_000}]}))
    assert with_tools > plain


def test_upto_requirements_forward_facilitator_kind_extra(manager, monkeypatch):
    async def _kind_extra(self):
        return {"facilitatorAddress": FACILITATOR, "newField": 1}

    monkeypatch.setattr(X402Manager, "get_upto_kind_extra", _kind_extra)
    req = asyncio.run(manager.build_requirements("upto", 0.01))
    assert req["extra"]["newField"] == 1
    assert req["extra"]["facilitatorAddress"] == FACILITATOR


def test_released_payment_can_be_reserved_again(monkeypatch):
    from src.x402 import release_payment, reserve_payment

    class _Redis(_FakeRedis):
        async def delete(self, key):
            self.store.pop(key, None)

    redis = _Redis()
    monkeypatch.setattr(x402, "get_redis", lambda: redis)
    payment = _signed()
    assert asyncio.run(reserve_payment(payment))
    asyncio.run(release_payment(payment))
    assert asyncio.run(reserve_payment(payment))


def test_expiry_read_only_from_the_scheme_field():
    upto = {**REQUIREMENTS, "scheme": "upto"}
    decoy = {"deadline": "5", "validBefore": "9999999999"}
    assert payment_valid_until({"accepted": upto, "payload": {"permit2Authorization": decoy}}) == 5
    assert (
        payment_valid_until({"accepted": REQUIREMENTS, "payload": {"authorization": {"deadline": "9999999999"}}})
        is None
    )


def test_max_output_tokens_is_priced(manager):
    body = {"messages": [], "max_tokens": 1, "max_output_tokens": 1_000_000}
    price, _ = asyncio.run(manager.compute_price("text", body))
    assert price == pytest.approx(2.0, rel=1e-3)


def test_supported_signers_are_cached_lowercase(monkeypatch, fresh_supported):
    monkeypatch.setattr(X402Manager, "_evm_signers", frozenset())
    kind = {"x402Version": 2, "scheme": "upto", "network": "eip155:8453", "extra": {"facilitatorAddress": "0xA"}}

    async def _get(url, headers=None):
        return _supported_response([kind], {"eip155:*": ["0xA", "0xB"], "solana:*": ["Sol1"]})

    monkeypatch.setattr(x402.client, "get", _get)
    mgr = X402Manager()
    asyncio.run(mgr.get_upto_kind_extra())
    assert mgr.facilitator_signers == {"0xa", "0xb"}


def test_any_facilitator_signer_is_accepted():
    required = {**REQUIREMENTS, "scheme": "upto", "extra": {**REQUIREMENTS["extra"], "facilitatorAddress": "0xA"}}
    other_signer = {**required, "extra": {**required["extra"], "facilitatorAddress": "0xB"}}
    assert accepted_matches(other_signer, required, facilitators={"0xa", "0xb"})
    assert not accepted_matches(other_signer, required)
    stranger = {**required, "extra": {**required["extra"], "facilitatorAddress": "0xC"}}
    assert not accepted_matches(stranger, required, facilitators={"0xa", "0xb"})
