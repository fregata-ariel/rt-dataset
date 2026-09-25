"""Job status and bundle derivation status routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from plateau_rt.viewer.api.errors import ApiError
from plateau_rt.viewer.api.routes_bundles import require_record
from plateau_rt.viewer.api.routes_derived import derived_payload
from plateau_rt.viewer.derive import DeriveNotFound, _find_cached, get_deriver
from plateau_rt.viewer.jobs import JobInfo, JobManager, eager_items, lookup_cached
from plateau_rt.viewer.store import Store

router = APIRouter()

_STATUSES = ("done", "queued", "running", "failed", "missing")


def _job_result(store: Store, info: JobInfo) -> dict[str, Any] | None:
    """Return the derived payload of a finished job, or None when unavailable."""
    if info.status != "done" or not info.deriver:
        return None
    try:
        deriver = get_deriver(info.deriver)
    except DeriveNotFound:
        return None
    cached = _find_cached(store, info.digest, info.member, deriver, info.params_key)
    if cached is None:
        return None
    return derived_payload(info.digest, cached)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Return one job's row plus its status URL and (when done) its result."""
    store: Store = request.app.state.store
    jobs: JobManager = request.app.state.jobs
    info = jobs.get(job_id)
    if info is None:
        raise ApiError(404, "not_found", f"unknown job {job_id!r}")
    payload = info.to_dict()
    payload["status_url"] = f"/api/jobs/{info.job_id}"
    payload["result"] = _job_result(store, info)
    return payload


def _eager_status(
    store: Store, jobs: JobManager, digest: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return the per-item eager status and the status counts of a bundle."""
    items: list[dict[str, Any]] = []
    counts = {status: 0 for status in _STATUSES}
    for item in eager_items(store, digest):
        params_key = item.prepared.params_key if item.prepared is not None else ""
        if item.prepared is None:
            status, job_id, reason, error = "failed", None, "error", item.error
        elif lookup_cached(store, item.prepared) is not None:
            status, job_id, reason, error = "done", None, None, None
        else:
            latest = jobs.latest(
                item.prepared.digest, item.member, item.deriver, item.prepared.params_key
            )
            if latest is None:
                status, job_id, reason, error = "missing", None, None, None
            elif latest.status == "done":
                status, job_id, reason, error = (
                    "missing",
                    latest.job_id,
                    latest.reason,
                    latest.error,
                )
            else:
                status, job_id, reason, error = (
                    latest.status,
                    latest.job_id,
                    latest.reason,
                    latest.error,
                )
        counts[status] = counts.get(status, 0) + 1
        items.append(
            {
                "member": item.member,
                "deriver": item.deriver,
                "params_key": params_key,
                "status": status,
                "job_id": job_id,
                "reason": reason,
                "error": error,
            }
        )
    counts["total"] = len(items)
    return items, counts


def _bundle_failures(store: Store, jobs: JobManager, digest: str) -> list[dict[str, Any]]:
    """Return every latest-per-key failed job of a bundle without a ready cache."""
    failures: list[dict[str, Any]] = []
    for (member, deriver, params_key), info in jobs.latest_by_key(digest).items():
        if info.status != "failed":
            continue
        try:
            registered = get_deriver(deriver)
        except DeriveNotFound:
            registered = None
        if registered is not None and (
            _find_cached(store, digest, member, registered, params_key) is not None
        ):
            continue
        failures.append(
            {
                "member": member,
                "deriver": deriver,
                "params_key": params_key,
                "kind": info.kind,
                "job_id": info.job_id,
                "reason": info.reason,
                "error": info.error,
                "updated_at": info.updated_at,
            }
        )
    return failures


@router.get("/bundles/{digest}/status")
def bundle_status(digest: str, request: Request) -> dict[str, Any]:
    """Return the eager completeness and failures of one bundle's derivations."""
    store: Store = request.app.state.store
    jobs: JobManager = request.app.state.jobs
    record = require_record(store, digest)
    items, counts = _eager_status(store, jobs, record.digest)
    complete = all(item["status"] == "done" for item in items)
    return {
        "digest": record.digest,
        "complete": complete,
        "eager": counts,
        "items": items,
        "failures": _bundle_failures(store, jobs, record.digest),
    }
