import base64
import json
import time

REQUIREMENTS = {
    "scheme": "exact",
    "network": "eip155:8453",
    "amount": "5000",
    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "payTo": "0x48E5A3e497D0C092D5B0Ba22B5A86ADA94227DFa",
    "maxTimeoutSeconds": 3600,
    "extra": {"name": "USD Coin", "version": "2"},
}


def payment_header(accepted: dict, **overrides) -> str:
    payload = {"signature": "0x", "authorization": {"validBefore": str(int(time.time()) + 3600)}}
    payment = {"x402Version": 2, "accepted": accepted, "payload": payload, **overrides}
    return base64.b64encode(json.dumps(payment).encode()).decode()
