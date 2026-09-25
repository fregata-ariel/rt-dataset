"""Deriver listing and derived-output routes."""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import FileResponse, JSONResponse, Response

from plateau_rt.viewer import safeio
from plateau_rt.viewer.api.errors import ApiError
from plateau_rt.viewer.api.routes_bundles import require_member, require_record
from plateau_rt.viewer.derive import (
    META_FILE,
    BadParams,
    DerivedResult,
    DeriveError,
    DeriveNotFound,
    Deriver,
    get_deriver,
    registered_derivers,
)
from plateau_rt.viewer.jobs import Prepared, lookup_cached, prepare
from plateau_rt.viewer.safeio import UnsafePathError
from plateau_rt.viewer.store import Store

router = APIRouter()

CACHE = "public, max-age=31536000, immutable"
META_MAX_BYTES = 16 * 1024 * 1024


def _quote(segment: str) -> str:
    """Percent-encode one URL path segment while keeping ``=`` and ``,`` literal."""
    return urllib.parse.quote(segment, safe="=,")


def derived_file_url(
    digest: str,
    member: str,
    deriver: str,
    version: int,
    params_key: str,
    links_key: str,
    name: str,
) -> str:
    """Return the versioned URL of one derived output file."""
    return (
        f"/api/bundles/{digest}/members/{_quote(member)}/derived/{_quote(deriver)}/"
        f"v{version}/{_quote(params_key)}/{_quote(links_key)}/{_quote(name)}"
    )


def _deriver_record(deriver: Deriver) -> dict[str, Any]:
    """Return the JSON description of one registered deriver."""
    spec = deriver.spec
    return {
        "name": spec.name,
        "version": spec.version,
        "kinds": list(spec.kinds),
        "eager": spec.eager,
        "params": [
            {
                "name": param.name,
                "kind": param.kind,
                "values": None if param.values is None else list(param.values),
                "min": param.min,
                "max": param.max,
                "step": param.step,
            }
            for param in spec.params
        ],
    }


def derived_payload(digest: str, result: DerivedResult) -> dict[str, Any]:
    """Return the JSON body describing one derivation result."""
    return {
        "deriver": result.deriver,
        "member": result.member,
        "version": result.version,
        "params": dict(result.params),
        "params_key": result.params_key,
        "links_key": result.links_key,
        "cached": result.cached,
        "files": [
            {
                "name": record.name,
                "size": record.size,
                "sha256": record.sha256,
                "url": derived_file_url(
                    digest,
                    result.member,
                    result.deriver,
                    result.version,
                    result.params_key,
                    result.links_key,
                    record.name,
                ),
            }
            for record in result.files
        ],
    }


def matches_etag(header: str | None, etag: str) -> bool:
    """Return True when an If-None-Match header matches ``etag`` or ``*``."""
    if header is None:
        return False
    for token in header.split(","):
        candidate = token.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate == etag or candidate == "*":
            return True
    return False


def _media_type(name: str) -> str:
    """Return the media type served for a derived file name."""
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".png"):
        return "image/png"
    return "application/octet-stream"


@router.get("/derivers")
def list_derivers() -> dict[str, Any]:
    """List every registered deriver with its parameters."""
    return {"derivers": [_deriver_record(deriver) for deriver in registered_derivers()]}


def _prepare_request(request: Request, digest: str, member: str, deriver: str) -> Prepared:
    """Parse the query parameters and prepare a derivation request."""
    store: Store = request.app.state.store
    record = require_record(store, digest)
    require_member(record, member)
    params: dict[str, str] = {}
    for key, value in request.query_params.multi_items():
        if key in params:
            raise ApiError(400, "bad_params", f"repeated parameter {key!r}", member)
        params[key] = value
    try:
        return prepare(store, record.digest, member, deriver, params)
    except DeriveNotFound as exc:
        raise ApiError(404, "not_found", str(exc), member) from exc
    except BadParams as exc:
        raise ApiError(400, "bad_params", str(exc), member) from exc
    except DeriveError as exc:
        raise ApiError(500, "derive_failed", str(exc), member) from exc


def _accepted(job: Any) -> JSONResponse:
    """Return the 202 body of a queued derivation job."""
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status_url": f"/api/jobs/{job.job_id}",
            "status": job.status,
        },
        status_code=202,
    )


@router.get("/bundles/{digest}/members/{member}/derived/{deriver}", response_model=None)
def derive_member(
    digest: str, member: str, deriver: str, request: Request
) -> dict[str, Any] | JSONResponse:
    """Return the cached outputs or queue a background derivation (202)."""
    store: Store = request.app.state.store
    prepared = _prepare_request(request, digest, member, deriver)
    cached = lookup_cached(store, prepared)
    if cached is not None:
        return derived_payload(prepared.digest, cached)
    job = request.app.state.jobs.submit(prepared, kind="lazy")
    return _accepted(job)


@router.post("/bundles/{digest}/members/{member}/derived/{deriver}/retry")
def retry_member(digest: str, member: str, deriver: str, request: Request) -> JSONResponse:
    """Re-queue a failed derivation; 409 when it is ready or not failed."""
    store: Store = request.app.state.store
    prepared = _prepare_request(request, digest, member, deriver)
    if lookup_cached(store, prepared) is not None:
        raise ApiError(409, "conflict", "derivation is already ready", member)
    jobs = request.app.state.jobs
    latest = jobs.latest(prepared.digest, member, prepared.deriver.spec.name, prepared.params_key)
    failed = latest is not None and latest.status == "failed"
    if latest is None:
        with store.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM derived WHERE digest = ? AND member = ? AND deriver = ? "
                "AND version = ? AND params_key = ? AND status = 'failed' LIMIT 1",
                (
                    prepared.digest,
                    member,
                    prepared.deriver.spec.name,
                    prepared.deriver.spec.version,
                    prepared.params_key,
                ),
            ).fetchone()
        failed = row is not None
    if not failed:
        status = latest.status if latest is not None else "none"
        raise ApiError(409, "conflict", f"derivation is not failed (status {status})", member)
    job = jobs.submit(prepared, kind="lazy", force=True)
    return _accepted(job)


@router.get(
    "/bundles/{digest}/members/{member}/derived/{deriver}"
    "/v{version:int}/{params_key}/{links_key}/{name}"
)
def get_derived_file(
    digest: str,
    member: str,
    deriver: str,
    version: int,
    params_key: str,
    links_key: str,
    name: str,
    request: Request,
) -> Response:
    """Serve one ready derived output file with ETag and immutable caching."""
    store: Store = request.app.state.store
    record = require_record(store, digest)
    require_member(record, member)
    try:
        registered_deriver = get_deriver(deriver)
    except DeriveNotFound as exc:
        raise ApiError(404, "not_found", str(exc), member) from exc
    if version != registered_deriver.spec.version:
        raise ApiError(
            404,
            "not_found",
            f"deriver {deriver!r} is at version {registered_deriver.spec.version}",
            member,
        )
    with store.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM derived WHERE digest = ? AND member = ? AND deriver = ? "
            "AND version = ? AND params_key = ? AND links_key = ? AND status = 'ready'",
            (record.digest, member, deriver, version, params_key, links_key),
        ).fetchone()
    if row is None:
        raise ApiError(404, "not_found", f"unknown derived file {name!r}", member)
    try:
        directory = safeio.resolve_inside(
            store.derived_dir(record.digest),
            "/".join([member, deriver, f"v{version}", params_key, links_key]),
        )
    except UnsafePathError as exc:
        raise ApiError(404, "not_found", str(exc), member) from exc
    try:
        meta = json.loads(safeio.read_bytes(directory / META_FILE, max_bytes=META_MAX_BYTES))
    except (OSError, ValueError) as exc:
        raise ApiError(404, "not_found", f"unknown derived file {name!r}", member) from exc
    entry = _find_meta_entry(meta, name)
    if entry is None:
        raise ApiError(404, "not_found", f"unknown derived file {name!r}", member)
    try:
        path = safeio.resolve_inside(directory, name)
    except UnsafePathError as exc:
        raise ApiError(404, "not_found", str(exc), member) from exc
    try:
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise ApiError(404, "not_found", f"unknown derived file {name!r}", member)
    except OSError as exc:
        raise ApiError(404, "not_found", f"unknown derived file {name!r}", member) from exc
    etag = f'"{entry["sha256"]}"'
    if matches_etag(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": CACHE})
    return FileResponse(
        path,
        media_type=_media_type(name),
        headers={
            "ETag": etag,
            "Cache-Control": CACHE,
            "X-Content-Type-Options": "nosniff",
        },
    )


def _find_meta_entry(meta: Any, name: str) -> dict[str, Any] | None:
    """Return the `_meta.json` file entry named ``name``, or None."""
    if not isinstance(meta, dict):
        return None
    files = meta.get("files")
    if not isinstance(files, list):
        return None
    for item in files:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None
