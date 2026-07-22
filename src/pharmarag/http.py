"""Shared HTTP client.

Exists because corporate/consumer TLS-inspecting proxies (Avast, Zscaler,
Kaspersky) re-sign every certificate, so httpx's bundled CA store rejects them.
Track B's graph builder makes ~9,000 RxNav calls — without this it fails on
call one.

Point PHARMARAG_CA_BUNDLE at your exported bundle, e.g.
    PHARMARAG_CA_BUNDLE=certs/ca-bundle-avast.pem
SSL_CERT_FILE and REQUESTS_CA_BUNDLE are honoured too, so an OS-level setting
already works without extra configuration. A relative value resolves against the
repo root, so it works regardless of the working directory uvicorn/streamlit
happen to start in.

Every outbound call — including the OpenAI SDK's — must go through this module.
The SDK builds its own httpx client otherwise, which uses the default CA store
and fails under an inspecting proxy even when everything else is configured.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def ca_bundle() -> str | bool:
    """Resolve a custom CA bundle, or True for httpx's default verification."""
    from pharmarag.config import ROOT

    for var in ("PHARMARAG_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(var)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return str(path)
    return True


def ssl_context() -> ssl.SSLContext | bool:
    """Verification context for the resolved bundle, or True for the default store.

    Avast's generated root sets basicConstraints non-critical, which RFC 5280 forbids
    for a CA. Python 3.13 turns VERIFY_X509_STRICT on by default in
    create_default_context(), so that root is rejected outright and every request fails
    with CERTIFICATE_VERIFY_FAILED — on 3.12 the same bundle verified fine. The flag is
    cleared only for an explicitly configured bundle: the operator has already chosen to
    trust that proxy, and the default store keeps strict checking.
    """
    bundle = ca_bundle()
    if not isinstance(bundle, str):
        return True
    ctx = ssl.create_default_context(cafile=bundle)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def client(*, timeout: float | httpx.Timeout = 30.0, **kwargs: Any) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        verify=ssl_context(),
        follow_redirects=True,
        headers={"User-Agent": "PharmaRAG/0.2 (portfolio project)"},
        **kwargs,
    )


def openai_client(*, timeout: float = 60.0, max_retries: int = 5) -> Any:
    """OpenAI SDK client sharing this module's TLS configuration.

    The timeout is set on the SDK **and** its http_client. Passing only a custom
    ``http_client`` is not enough: the OpenAI SDK sends its own per-request timeout
    (default ~10 min) which overrides the httpx client's, so a connection stalled by
    a TLS-inspecting proxy (Avast) can hang for the SDK default times its retries — an hour
    in practice. Setting ``timeout`` here makes a stalled read fail fast (60s) and
    ``max_retries`` lets the SDK reconnect on the next attempt, which normally
    succeeds. A short connect timeout surfaces a dead proxy immediately.
    """
    from openai import OpenAI

    from pharmarag.config import settings

    read_timeout = httpx.Timeout(timeout, connect=10.0)
    return OpenAI(
        api_key=settings.openai_api_key,
        http_client=client(timeout=read_timeout),
        timeout=read_timeout,
        max_retries=max_retries,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get_json(url: str, params: dict[str, Any] | None = None, *, timeout: float = 30.0) -> Any:
    """GET with exponential backoff. RxNav rate-limits; retrying is expected."""
    with client(timeout=timeout) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()
