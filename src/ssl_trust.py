import glob
import os
import ssl

import certifi


def build_ssl_context() -> ssl.SSLContext:
    """Trust the public CAs plus any pinned self-signed upstream certs in ../certs/*.crt."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    certs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "certs")
    for crt in sorted(glob.glob(os.path.join(certs_dir, "*.crt"))):
        ctx.load_verify_locations(cafile=crt)
    return ctx


# Shared context for every httpx client that may reach a pinned self-signed upstream
# (proxy forwarding and health checks).
SSL_CONTEXT = build_ssl_context()
