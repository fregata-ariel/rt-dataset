"""Build-hashed immutable static assets for the viewer frontend."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse, Response

from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.api.errors import ApiError
from plateau_rt.viewer.api.routes_derived import matches_etag

BUILD_PLACEHOLDER = "__BUILD__"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"


@dataclass(frozen=True)
class StaticAsset:
    """One in-memory static file with its sha256 and media type."""

    body: bytes
    sha256: str
    media_type: str


@dataclass(frozen=True)
class StaticBundle:
    """Snapshot of the static directory with its build id."""

    build: str
    files: Mapping[str, StaticAsset]
    index_html: bytes | None


def media_type_for(relpath: str) -> str:
    """Return the media type served for a static relative path."""
    name = relpath.rsplit("/", 1)[-1]
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix in ("js", "mjs"):
        return "text/javascript; charset=utf-8"
    if suffix == "css":
        return "text/css; charset=utf-8"
    if suffix == "json":
        return "application/json"
    if suffix == "map":
        return "application/json"
    if suffix == "html":
        return "text/html; charset=utf-8"
    if suffix == "svg":
        return "image/svg+xml"
    if suffix == "png":
        return "image/png"
    if suffix == "woff2":
        return "font/woff2"
    if suffix in ("txt", "md") or suffix == "":
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def _included(relpath: str) -> bool:
    """Return True when a POSIX relpath is eligible for the static bundle."""
    parts = relpath.split("/")
    for part in parts:
        if part.startswith("."):
            return False
        if part == "__pycache__":
            return False
    return True


def load_static(directory: Path) -> StaticBundle | None:
    """Snapshot ``directory`` into memory, or None when it is not a directory."""
    if not directory.is_dir():
        return None
    raw: dict[str, bytes] = {}
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs.sort()
        files.sort()
        for name in files:
            full = Path(root) / name
            relpath = full.relative_to(directory).as_posix()
            if not _included(relpath):
                continue
            if os.path.islink(full):
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            raw[relpath] = full.read_bytes()
    digests = {relpath: hashlib.sha256(body).hexdigest() for relpath, body in raw.items()}
    lines = "".join(f"{relpath}\t{digests[relpath]}\n" for relpath in sorted(digests))
    build = hashlib.sha256(lines.encode("utf-8")).hexdigest()[:12]
    index_html: bytes | None = None
    if "index.html" in raw:
        index_html = raw["index.html"].replace(
            BUILD_PLACEHOLDER.encode("utf-8"), build.encode("utf-8")
        )
    assets: dict[str, StaticAsset] = {}
    for relpath, body in raw.items():
        if relpath == "index.html":
            continue
        assets[relpath] = StaticAsset(
            body=body, sha256=digests[relpath], media_type=media_type_for(relpath)
        )
    return StaticBundle(build=build, files=assets, index_html=index_html)


def _serve_asset(asset: StaticAsset, *, immutable: bool, etag: str) -> Response:
    """Return a 200 response for one static asset."""
    cache = IMMUTABLE_CACHE if immutable else NO_CACHE
    return Response(
        asset.body,
        media_type=asset.media_type,
        headers={
            "ETag": etag,
            "Cache-Control": cache,
            "X-Content-Type-Options": "nosniff",
        },
    )


def install_static_routes(app: FastAPI, directory: Path) -> StaticBundle | None:
    """Register ``/`` and ``/static/{path}`` on ``app`` from a snapshot of ``directory``."""
    bundle = load_static(directory)
    app.state.static_bundle = bundle
    app.state.static_build = bundle.build if bundle is not None else None

    @app.get("/")
    def index() -> Response:
        """Serve the frontend index when installed, else a plain-text description."""
        if bundle is not None and bundle.index_html is not None:
            return Response(
                bundle.index_html,
                media_type="text/html; charset=utf-8",
                headers={"Cache-Control": NO_CACHE, "X-Content-Type-Options": "nosniff"},
            )
        return PlainTextResponse(
            f"plateau_rt viewer backend {VIEWER_VERSION}: no frontend installed. "
            "The HTTP API is under /api (try /api/health); see docs/viewer.md.",
            headers={"Cache-Control": NO_CACHE},
        )

    @app.get("/static/{path:path}")
    def serve_static(path: str, request: Request) -> Response:
        """Serve one snapshotted static file by dictionary lookup only."""
        if bundle is None:
            raise ApiError(404, "not_found", "unknown static file")
        first, _, rest = path.partition("/")
        asset: StaticAsset | None = None
        immutable = False
        if first == bundle.build and rest:
            asset = bundle.files.get(rest)
            immutable = asset is not None
        if asset is None:
            asset = bundle.files.get(path)
            immutable = False
        if asset is None:
            raise ApiError(404, "not_found", "unknown static file")
        etag = f'"{asset.sha256}"'
        if matches_etag(request.headers.get("if-none-match"), etag):
            cache = IMMUTABLE_CACHE if immutable else NO_CACHE
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache})
        return _serve_asset(asset, immutable=immutable, etag=etag)

    return bundle
