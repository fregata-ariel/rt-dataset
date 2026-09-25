"""Bundle upload, listing, metadata, deletion and raw file routes."""

from __future__ import annotations

import errno
import json
import logging
import os
import posixpath
import re
import shutil
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from plateau_rt.viewer import safeio
from plateau_rt.viewer.api.errors import ApiError
from plateau_rt.viewer.extract import UnsafeArchiveError
from plateau_rt.viewer.ingest import ingest_archive
from plateau_rt.viewer.kinds import BundleValidationError, UnknownKindError
from plateau_rt.viewer.safeio import UnsafePathError
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import BundleRecord, Store

router = APIRouter()

_LOGGER = logging.getLogger(__name__)

DELETE_BODY_LIMIT = 4096
_QUERY_NAME_RE = re.compile(r".{1,200}\Z", re.DOTALL)
_CONTENT_LENGTH_RE = re.compile(r"[0-9]+\Z")
_UPLOAD_CONTENT_TYPES = ("application/octet-stream",)
_FILENAME_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_DISK_FULL_ERRNOS = (errno.ENOSPC, errno.EDQUOT)


def require_record(store: Store, digest: str) -> BundleRecord:
    """Return the bundle record for ``digest`` or raise a 404 ``ApiError``."""
    try:
        record = store.get(digest)
    except ValueError as exc:
        raise ApiError(404, "not_found", f"unknown bundle {digest!r}") from exc
    if record is None:
        raise ApiError(404, "not_found", f"unknown bundle {digest!r}")
    return record


def require_member(record: BundleRecord, member: str) -> dict[str, Any]:
    """Return the member record with id ``member`` or raise a 404 ``ApiError``."""
    for info in record.members:
        if info.get("id") == member:
            return info
    raise ApiError(404, "not_found", f"unknown member {member!r}", member)


def summary(record: BundleRecord) -> dict[str, Any]:
    """Return the JSON summary of one bundle record."""
    return {
        "digest": record.digest,
        "name": record.name,
        "created_at": record.created_at,
        "total_bytes": record.total_bytes,
        "file_count": record.file_count,
        "status": record.status,
        "members": list(record.members),
    }


def _valid_name(name: str) -> bool:
    """Return True for a 1..200 character name without control characters."""
    if _QUERY_NAME_RE.fullmatch(name) is None:
        return False
    return all(ord(char) >= 32 and ord(char) != 127 for char in name)


def _content_length(request: Request, settings: ViewerSettings) -> int | None:
    """Validate the Content-Length header (if any) against the upload cap."""
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    if _CONTENT_LENGTH_RE.fullmatch(raw) is None:
        raise ApiError(400, "bad_params", "invalid Content-Length header")
    length = int(raw)
    if length > settings.max_upload_bytes:
        raise ApiError(413, "too_large", "upload exceeds max_upload_bytes")
    return length


async def _read_upload(request: Request, settings: ViewerSettings, upload_path: Path) -> None:
    """Stream the request body into ``upload_path`` enforcing the upload cap."""
    written = 0
    fd = os.open(upload_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        async for chunk in request.stream():
            if written + len(chunk) > settings.max_upload_bytes:
                raise ApiError(413, "too_large", "upload exceeds max_upload_bytes")
            handle.write(chunk)
            written += len(chunk)


def _map_ingest_error(exc: BaseException) -> ApiError:
    """Translate an ingest exception into the documented HTTP error."""
    if isinstance(exc, UnsafeArchiveError):
        return ApiError(400, "unsafe_archive", str(exc))
    if isinstance(exc, UnknownKindError):
        return ApiError(400, "unknown_kind", exc.message, exc.member_id)
    if isinstance(exc, BundleValidationError):
        return ApiError(400, "validation", exc.message, exc.member_id)
    if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
        return ApiError(507, "insufficient_storage", str(exc))
    raise exc


@router.put("/bundles/upload")
async def upload_bundle(request: Request) -> JSONResponse:
    """Ingest a raw archive body into the store."""
    store: Store = request.app.state.store
    settings = store.settings
    name = request.query_params.get("name", "bundle")
    if not _valid_name(name):
        raise ApiError(400, "bad_params", "invalid bundle name")
    content_type = request.headers.get("content-type")
    if content_type is not None:
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in _UPLOAD_CONTENT_TYPES:
            raise ApiError(
                400,
                "bad_params",
                "Content-Type must be application/octet-stream",
            )
    _content_length(request, settings)
    if shutil.disk_usage(store.data_dir).free < settings.max_extracted_bytes:
        raise ApiError(507, "insufficient_storage", "free space is below max_extracted_bytes")
    upload_path = store.staging_dir / f"{uuid.uuid4().hex}.upload"
    try:
        await _read_upload(request, settings, upload_path)
        result = await run_in_threadpool(ingest_archive, store, upload_path, name=name)
        if result.created:
            try:
                await run_in_threadpool(request.app.state.on_bundle_committed, store, result.digest)
            except Exception:
                _LOGGER.exception("on_bundle_committed hook failed for %s", result.digest)
        return JSONResponse(
            {
                "digest": result.digest,
                "created": result.created,
                "members": list(result.members),
            },
            status_code=201 if result.created else 200,
        )
    except ApiError:
        raise
    except ClientDisconnect as exc:
        raise ApiError(400, "bad_params", "client disconnected") from exc
    except BaseException as exc:
        mapped = _map_ingest_error(exc)
        raise mapped from exc
    finally:
        try:
            os.unlink(upload_path)
        except FileNotFoundError:
            pass


@router.get("/bundles")
def list_bundles(request: Request) -> dict[str, Any]:
    """List every stored bundle with its summary."""
    store: Store = request.app.state.store
    return {"bundles": [summary(record) for record in store.list()]}


@router.get("/bundles/{digest}")
def get_bundle(digest: str, request: Request) -> dict[str, Any]:
    """Return one bundle's summary, error, validated_with and derived rows."""
    store: Store = request.app.state.store
    record = require_record(store, digest)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT member, deriver, version, params_key, links_key, status, error, updated_at "
            "FROM derived WHERE digest = ? "
            "ORDER BY member, deriver, version, params_key, links_key",
            (record.digest,),
        ).fetchall()
    payload = summary(record)
    payload["error"] = record.error
    payload["validated_with"] = record.validated_with
    payload["derived"] = [
        {
            "member": row["member"],
            "deriver": row["deriver"],
            "version": row["version"],
            "params_key": row["params_key"],
            "links_key": row["links_key"],
            "status": row["status"],
            "error": row["error"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]
    return payload


@router.delete("/bundles/{digest}")
async def delete_bundle(digest: str, request: Request) -> dict[str, Any]:
    """Delete one bundle after a body confirming its digest."""
    store: Store = request.app.state.store
    record = require_record(store, digest)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > DELETE_BODY_LIMIT:
            raise ApiError(413, "too_large", "request body exceeds 4096 bytes")
    try:
        payload = json.loads(bytes(body))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(400, "bad_params", "confirmation must equal the bundle digest") from exc
    if not isinstance(payload, dict) or payload.get("confirm") != record.digest:
        raise ApiError(400, "bad_params", "confirmation must equal the bundle digest")
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is not None:
        await run_in_threadpool(jobs.cancel_bundle, record.digest)
    store.delete(record.digest)
    return {"digest": record.digest, "deleted": True}


def _member_directory(info: dict[str, Any]) -> str:
    """Return the bundle-relative directory holding a member's files."""
    member_path = str(info.get("path", ""))
    if info.get("kind") == "scene":
        return posixpath.dirname(member_path) or "."
    return member_path or "."


def _content_disposition(basename: str) -> str:
    """Build the ASCII + RFC 5987 Content-Disposition value for ``basename``."""
    ascii_name = "".join(char if char in _FILENAME_SAFE else "_" for char in basename)
    quoted = urllib.parse.quote(basename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quoted}"


@router.get("/bundles/{digest}/members/{member}/raw/{path:path}")
def get_raw_file(digest: str, member: str, path: str, request: Request) -> FileResponse:
    """Serve one file below a member's directory from the read-only raw tree."""
    store: Store = request.app.state.store
    record = require_record(store, digest)
    info = require_member(record, member)
    member_dir = _member_directory(info)
    raw_root = store.raw_dir(record.digest)
    try:
        member_root = (
            raw_root if member_dir in ("", ".") else safeio.resolve_inside(raw_root, member_dir)
        )
        target = safeio.resolve_inside(member_root, path)
    except UnsafePathError:
        raise ApiError(400, "bad_params", "invalid path", member) from None
    if not path or not os.path.isfile(target) or os.path.isdir(target):
        raise ApiError(404, "not_found", f"unknown file {path!r}", member)
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": _content_disposition(target.name),
        "Content-Security-Policy": "sandbox",
        "Cache-Control": "private, max-age=0",
    }
    return FileResponse(target, media_type="application/octet-stream", headers=headers)
