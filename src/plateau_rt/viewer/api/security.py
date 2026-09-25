"""App-wide security headers, CSRF and Host allow-list middleware (Sionna-free)."""

from __future__ import annotations

import urllib.parse
from collections.abc import Collection, Mapping

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from plateau_rt.viewer.api import errors
from plateau_rt.viewer.settings import ViewerSettings

CONTENT_SECURITY_POLICY: str = (
    "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'"
)
SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
CSRF_HEADER: str = "X-Viewer-Request"
CSRF_HEADER_VALUE: str = "1"
SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


def merge_csp(existing: str | None) -> str:
    """Return the baseline CSP with a non-empty route policy appended."""
    if existing is None:
        return CONTENT_SECURITY_POLICY
    stripped = existing.strip()
    if not stripped:
        return CONTENT_SECURITY_POLICY
    return f"{CONTENT_SECURITY_POLICY}; {stripped}"


def origin_allowed(origin: str, host: str | None, allowed_origins: Collection[str]) -> bool:
    """Return True when ``origin`` is same-origin with ``host`` or explicitly allowed."""
    normalized = origin.strip().lower()
    if not normalized or normalized == "null":
        return False
    parts = urllib.parse.urlsplit(normalized)
    if parts.scheme not in ("http", "https"):
        return False
    if not parts.netloc or "@" in parts.netloc:
        return False
    if parts.path or parts.query or parts.fragment:
        return False
    candidate = f"{parts.scheme}://{parts.netloc}"
    if candidate in allowed_origins:
        return True
    return host is not None and parts.netloc == host.strip().lower()


def csrf_violation(
    method: str,
    headers: Mapping[str, str] | Headers,
    allowed_origins: Collection[str],
) -> str | None:
    """Return an English violation message, or None when the request may proceed."""
    if method.upper() in SAFE_METHODS:
        return None
    normalized = headers if isinstance(headers, Headers) else Headers(headers)
    value = normalized.get(CSRF_HEADER)
    if value is None or value.strip() != CSRF_HEADER_VALUE:
        return f"missing {CSRF_HEADER}: {CSRF_HEADER_VALUE} header"
    origin = normalized.get("origin")
    if origin is not None and not origin_allowed(origin, normalized.get("host"), allowed_origins):
        return f"cross-origin request from {origin!r} is not allowed"
    return None


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding the baseline security headers to HTTP responses."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the wrapped ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap ``send`` on HTTP scopes so every response carries the headers."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                headers["content-security-policy"] = merge_csp(
                    headers.get("content-security-policy")
                )
                headers["x-content-type-options"] = "nosniff"
                headers["referrer-policy"] = "no-referrer"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            # Starlette's ServerErrorMiddleware is always outermost, so its 500 would
            # bypass this middleware. Answer the 500 here (with the headers) and re-raise
            # so the server still logs the traceback; the outer middleware then sees a
            # started response and sends nothing more.
            if not started:
                response = PlainTextResponse("Internal Server Error", status_code=500)
                await response(scope, receive, send_with_headers)
            raise


class CSRFMiddleware:
    """Pure ASGI middleware requiring the CSRF header on unsafe HTTP methods."""

    def __init__(self, app: ASGIApp, *, allowed_origins: Collection[str] = ()) -> None:
        """Store the wrapped app and the allowed cross-origins."""
        self.app = app
        self.allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Answer 403 without reading the body when the request violates CSRF rules."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        message = csrf_violation(scope["method"], Headers(scope=scope), self.allowed_origins)
        if message is not None:
            await errors.error_response(403, "forbidden", message)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def install_security(app: FastAPI, settings: ViewerSettings) -> None:
    """Install security headers, trusted hosts and CSRF middleware in order."""
    app.add_middleware(CSRFMiddleware, allowed_origins=frozenset(settings.allowed_origins))
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(settings.allowed_hosts),
        www_redirect=False,
    )
    app.add_middleware(SecurityHeadersMiddleware)
