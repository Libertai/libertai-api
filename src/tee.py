"""Reaching a confidential V-PROGRAM, which PKI cannot describe.

A `tee://<item-hash>` entry in MODELS_CONFIG names an Aleph V-PROGRAM instead
of an address. Its address is discovered from the item hash, and its TLS
certificate is trusted because an AMD-signed attestation report proves it
belongs to the published enclave -- not because a certificate authority vouched
for it.

A pinned `certs/*.crt` cannot do that job here. The guest mints a fresh
self-signed certificate on every boot whose only SAN is `localhost`, and the
node forwards a different host port each time it starts, so there is nothing
stable for a CA check to bind to. The measurement is the stable identity.

Skipping verification instead is not an option: `create_signed_payload` signs
the API key list but does not encrypt it, so it travels as cleartext inside
TLS. An endpoint that cannot be proved therefore gets no request at all.

Verification lives in `libertai-confidential-inference`, the same package a
client installs to verify this deployment -- so the proxy runs the checks its
users run, rather than a second implementation that could drift. The import is
still guarded: without the package a `tee://` entry is refused rather than
contacted.
"""

from __future__ import annotations

import asyncio

import httpx

from src.logger import setup_logger

logger = setup_logger(__name__)

SCHEME = "tee://"

# Published measurements are read per deployment, and verification needs a VCEK
# from AMD, so proving an endpoint takes a couple of seconds and blocks. It is
# done once per enclave and cached until the enclave stops answering.
_clients: dict[str, tuple[str, httpx.AsyncClient]] = {}
_locks: dict[str, asyncio.Lock] = {}

try:
    from libertai_confidential import resolve_deployment
    from libertai_confidential.transport import attested_ssl_context

    AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the deployment's extras
    resolve_deployment = None  # type: ignore[assignment]
    attested_ssl_context = None  # type: ignore[assignment]
    AVAILABLE = False


def is_tee(server: str) -> bool:
    return server.startswith(SCHEME)


def item_hash(server: str) -> str:
    return server[len(SCHEME) :]


def _verify(hash_: str) -> tuple[str, httpx.AsyncClient]:
    """Prove one enclave and build a client pinned to the certificate it proved.

    Runs off the event loop: resolving the deployment and fetching the VCEK are
    blocking calls.
    """
    try:
        deployment = resolve_deployment(hash_)
    except Exception as e:
        # The package raises its own exception types, which derive from
        # Exception and nothing narrower. Callers of this module handle
        # RuntimeError, so anything else escaping here would abort a whole
        # health sweep rather than marking one endpoint down.
        raise RuntimeError(f"cannot resolve {hash_[:12]}: {type(e).__name__}: {e}") from e

    failures = []
    for origin in deployment.candidates:
        try:
            context, measurement = attested_ssl_context(origin, deployment.measurements)
        except Exception as e:
            # Every candidate is worth trying: an address that cannot be
            # reached from here says nothing about the next one.
            failures.append(f"{origin}: {type(e).__name__}: {e}")
            continue
        logger.info(f"verified enclave {hash_[:12]} at {origin} (measurement {measurement[:16]}...)")
        # The context trusts exactly the certificate that was proved, so a
        # restarted enclave -- which mints a new one -- fails here and is
        # re-verified rather than silently trusted.
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=300.0)
        # trust_env=False keeps this off HTTP_PROXY: the node forwards a fresh
        # host port on every boot, and the proxy only tunnels CONNECT to ports
        # on its allow-list. Nothing is lost by going direct, since the context
        # already pins the certificate that was attested.
        return origin, httpx.AsyncClient(verify=context, timeout=30.0, limits=limits, trust_env=False)
    raise RuntimeError(f"no attested endpoint for {hash_[:12]}: {'; '.join(failures)}")


async def target(server: str, default: httpx.AsyncClient) -> tuple[str, httpx.AsyncClient]:
    """The base URL and client to use for a configured server entry.

    Anything that is not a `tee://` entry is returned untouched, on the shared
    client.
    """
    if not is_tee(server):
        return server, default
    if not AVAILABLE:
        raise RuntimeError(f"{server} needs the libertai-confidential-inference package, which is not installed")

    hash_ = item_hash(server)
    cached = _clients.get(hash_)
    if cached is not None:
        return cached

    lock = _locks.setdefault(hash_, asyncio.Lock())
    async with lock:
        # Another request may have verified it while this one waited.
        cached = _clients.get(hash_)
        if cached is not None:
            return cached
        resolved = await asyncio.to_thread(_verify, hash_)
        _clients[hash_] = resolved
        return resolved


async def invalidate(server: str) -> None:
    """Forget a proved endpoint, so the next call re-verifies it.

    Called when a request to it fails: the usual cause is a reboot, which moves
    the port and changes the certificate.
    """
    if not is_tee(server):
        return
    entry = _clients.pop(item_hash(server), None)
    if entry is not None:
        await entry[1].aclose()


async def close_all() -> None:
    for _base, client in _clients.values():
        await client.aclose()
    _clients.clear()
