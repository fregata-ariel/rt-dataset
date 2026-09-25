"""Shared TestClient factory for the viewer API tests (allowed Host + CSRF header)."""

from __future__ import annotations

import warnings
from typing import Any

CLIENT_BASE_URL = "http://127.0.0.1"
CLIENT_HEADERS = {"X-Viewer-Request": "1"}

with warnings.catch_warnings():
    # Starlette 1.x warns that its TestClient will move from httpx to httpx2.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient


def viewer_client(app: Any, **kwargs: Any) -> TestClient:
    """Return a TestClient on an allowed Host that sends the CSRF header."""
    kwargs.setdefault("base_url", CLIENT_BASE_URL)
    headers = dict(CLIENT_HEADERS)
    headers.update(kwargs.pop("headers", None) or {})
    return TestClient(app, headers=headers, **kwargs)
