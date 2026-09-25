"""Tests for the viewer job manager and its HTTP routes (plateau_rt.viewer.jobs)."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from viewer_bundle_fixtures import BundleFixture, make_archive, write_fixture_bundle
from viewer_client import viewer_client
from viewer_job_derivers import (
    MARKS_ENV,
    Boom,
    EagerSlowDeriver,
    EnumDeriver,
    FlakyDeriver,
    MemDeriver,
    SlowDeriver,
    read_marks,
)

from plateau_rt.viewer.api import create_app
from plateau_rt.viewer.derive import BadParams, DeriveNotFound, registered
from plateau_rt.viewer.ingest import ingest_staged
from plateau_rt.viewer.jobs import JobManager, prepare
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

with warnings.catch_warnings():
    # Starlette 1.x warns that its TestClient will move from httpx to httpx2.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

MiB = 1 << 20
GiB = 1 << 30
OCTET = {"Content-Type": "application/octet-stream"}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    """Build the canonical v3 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("jobs-bundle") / "bundle"
    return write_fixture_bundle(root, schema_version=3)


@pytest.fixture(scope="module")
def zip_bytes(bundle: BundleFixture, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build a zip of the fixture bundle once per module."""
    out = tmp_path_factory.mktemp("jobs-zip") / "bundle.zip"
    make_archive(bundle.root, out, "zip")
    return out.read_bytes()


def _settings(tmp_path: Path, **overrides: Any) -> ViewerSettings:
    """Return viewer settings rooted at ``tmp_path`` with fast, hermetic defaults."""
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "max_upload_bytes": 16 * MiB,
        "max_extracted_bytes": 64 * MiB,
        "max_files": 1000,
    }
    base.update(overrides)
    return ViewerSettings(**base)


def make_app(tmp_path: Path, **overrides: Any) -> Any:
    """Build an app with no static directory."""
    return create_app(_settings(tmp_path, **overrides), static_dir=tmp_path / "nostatic")


def upload(client: TestClient, data: bytes, name: str = "fixture") -> Any:
    """PUT an archive body as an octet-stream upload."""
    return client.put(f"/api/bundles/upload?name={name}", content=data, headers=OCTET)


def poll_job(client: TestClient, job_id: str, timeout: float = 60) -> dict[str, Any]:
    """Poll one job until it is terminal and return its JSON body."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout} s")


def _marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a marks directory and point the derivers at it."""
    directory = tmp_path / "marks"
    directory.mkdir()
    monkeypatch.setenv(MARKS_ENV, str(directory))
    return directory


def _wait_for_mark(directory: Path, prefix: str, timeout: float = 30) -> dict[str, Any]:
    """Wait for a mark file with ``prefix`` to appear and return its parsed contents."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = list(directory.glob(f"{prefix}-*.json"))
        if files:
            return json.loads(files[0].read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"mark {prefix} never appeared")


def _assert_pid_gone(pid: int, timeout: float = 2) -> None:
    """Assert that ``pid`` no longer exists within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} is still alive")


def _insert_job(
    store: Store,
    *,
    job_id: str,
    status: str,
    kind: str,
    digest: str,
    member: str,
    deriver: str,
    params_key: str,
    error: str | None = None,
) -> None:
    """Insert one jobs row directly for restart and parsing tests."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(job_id, kind, digest, member, deriver, params_key, status, stage, "
            "done_bytes, total_bytes, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)",
            (
                job_id,
                kind,
                digest,
                member,
                deriver,
                params_key,
                status,
                "starting",
                error,
                now,
                now,
            ),
        )


def _count_jobs(store: Store) -> int:
    """Return the number of rows in the jobs table."""
    with store.connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])


def test_slow_deriver_202_then_200_and_dedup(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache miss answers 202, deduplicates concurrent requests and then serves 200."""
    marks = _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path)
    with registered(SlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tslow"
            params = {"ms": "1500", "tag": "1"}

            async def fire() -> list[httpx.Response]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://127.0.0.1",
                    headers={"X-Viewer-Request": "1"},
                ) as async_client:
                    return await asyncio.gather(
                        *[async_client.get(url, params=params) for _ in range(5)]
                    )

            responses = asyncio.run(fire())
            assert all(response.status_code == 202 for response in responses)
            job_ids = {response.json()["job_id"] for response in responses}
            assert len(job_ids) == 1
            job_id = job_ids.pop()
            for response in responses:
                body = response.json()
                assert body["status_url"] == f"/api/jobs/{job_id}"
                assert body["status"] in ("queued", "running")
            store: Store = app.state.store
            with store.connect() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE deriver = 'tslow'"
                ).fetchone()[0]
            assert count == 1

            info = poll_job(client, job_id)
            assert info["status"] == "done"
            assert info["stage"] == "done"
            assert [item["name"] for item in info["result"]["files"]] == ["out.json"]

            again = client.get(url, params=params)
            assert again.status_code == 200
            assert again.json()["cached"] is True
            file_url = again.json()["files"][0]["url"]
            served = client.get(file_url)
            assert served.status_code == 200
            assert served.json() == {"tag": 1}
            assert len(list(marks.glob("tslow-1-*.json"))) == 1


def test_timeout_kills_child(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that overruns the timeout is killed and the job fails with reason timeout."""
    marks = _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path, derive_timeout_s=3)
    with registered(SlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tslow"
            response = client.get(url, params={"ms": "60000", "tag": "2"})
            assert response.status_code == 202
            info = poll_job(client, response.json()["job_id"], timeout=30)
            assert info["status"] == "failed"
            assert info["reason"] == "timeout"
            mark = _wait_for_mark(marks, "tslow-2")
            assert mark["end"] is None
            _assert_pid_gone(int(mark["pid"]))
            assert client.get("/api/health").status_code == 200


def test_memory_limit(tmp_path: Path, zip_bytes: bytes) -> None:
    """The child's address-space limit is enforced without killing the server."""
    parent_pid = os.getpid()
    app = make_app(tmp_path, derive_mem_bytes=2 * GiB)
    with registered(MemDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tmem"
            ok = client.get(url, params={"mib": "64"})
            assert ok.status_code == 202
            assert poll_job(client, ok.json()["job_id"])["status"] == "done"
            bad = client.get(url, params={"mib": "3072"})
            assert bad.status_code == 202
            info = poll_job(client, bad.json()["job_id"], timeout=60)
            assert info["status"] == "failed"
            assert info["reason"] == "memory_limit"
            assert client.get("/api/health").status_code == 200
            assert os.getpid() == parent_pid


@pytest.mark.parametrize(("workers", "expect_overlap"), [(1, False), (2, True)])
def test_concurrency_limit(
    tmp_path: Path,
    zip_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    expect_overlap: bool,
) -> None:
    """One worker serialises derivations; two workers overlap them."""
    marks = _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path, max_concurrent_derives=workers)
    app.state.on_bundle_committed = lambda store, digest: None
    with registered(SlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tslow"
            job_ids = []
            for tag in (10, 11, 12):
                response = client.get(url, params={"ms": "400", "tag": str(tag)})
                assert response.status_code == 202
                job_ids.append(response.json()["job_id"])
            for job_id in job_ids:
                poll_job(client, job_id)
    finished = [
        mark
        for mark in read_marks(marks)
        if mark["deriver"] == "tslow" and mark.get("end") is not None
    ]
    assert len(finished) == 3
    intervals = sorted((mark["start"], mark["end"]) for mark in finished)
    overlaps = any(
        intervals[index + 1][0] < intervals[index][1] for index in range(len(intervals) - 1)
    )
    assert overlaps is expect_overlap


def test_lazy_runs_before_eager(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newly queued lazy job is promoted ahead of pending eager jobs."""
    marks = _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path, max_concurrent_derives=1)
    app.state.on_bundle_committed = lambda store, digest: None
    with registered(SlowDeriver(), EagerSlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            base = f"/api/bundles/{digest}/members/dataset/derived"
            jobs: JobManager = app.state.jobs
            blocker = client.get(f"{base}/tslow", params={"ms": "800", "tag": "20"})
            assert blocker.status_code == 202
            eager_jobs = jobs.submit_eager(digest)
            assert {job.deriver for job in eager_jobs} == {"overview", "teagerslow"}
            lazy = client.get(f"{base}/tslow", params={"ms": "10", "tag": "21"})
            assert lazy.status_code == 202
            job_ids = [blocker.json()["job_id"], lazy.json()["job_id"]]
            job_ids.extend(job.job_id for job in eager_jobs)
            jobs.wait(job_ids, timeout=60)
    finished = [mark for mark in read_marks(marks) if mark.get("end") is not None]
    tag21 = next(mark for mark in finished if mark["deriver"] == "tslow" and mark["tag"] == 21)
    eager = next(mark for mark in finished if mark["deriver"] == "teagerslow")
    assert tag21["start"] < eager["start"]


def test_retry_flow(tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed derivations are sticky; retry re-runs them and conflicts once ready."""
    marks = _marks(tmp_path, monkeypatch)
    (marks / "flaky-fail").write_text("", encoding="utf-8")
    app = make_app(tmp_path)
    with registered(FlakyDeriver(), EnumDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            base = f"/api/bundles/{digest}/members/dataset/derived"
            first = client.get(f"{base}/tflaky")
            assert first.status_code == 202
            job_id = first.json()["job_id"]
            info = poll_job(client, job_id)
            assert info["status"] == "failed"
            assert info["reason"] == "error"
            assert "flaky failure" in info["error"]
            sticky = client.get(f"{base}/tflaky")
            assert sticky.status_code == 202
            assert sticky.json()["job_id"] == job_id
            (marks / "flaky-fail").unlink()
            retry = client.post(f"{base}/tflaky/retry")
            assert retry.status_code == 202
            new_id = retry.json()["job_id"]
            assert new_id != job_id
            assert poll_job(client, new_id)["status"] == "done"
            ready = client.get(f"{base}/tflaky")
            assert ready.status_code == 200
            assert ready.json()["cached"] is True
            conflict = client.post(f"{base}/tflaky/retry")
            assert conflict.status_code == 409
            assert conflict.json()["error"]["type"] == "conflict"
            never = client.post(f"{base}/tenum/retry", params={"mode": "a"})
            assert never.status_code == 409
            assert never.json()["error"]["type"] == "conflict"
            bad = client.post(f"{base}/tenum/retry", params={"mode": "zzz"})
            assert bad.status_code == 400
            assert bad.json()["error"]["type"] == "bad_params"
            assert client.post(f"{base}/nope/retry").status_code == 404


def test_retry_while_active_conflicts(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying a queued or running derivation answers 409 and starts no second job."""
    _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path)
    app.state.on_bundle_committed = lambda store, digest: None
    with registered(SlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tslow"
            params = {"ms": "2000", "tag": "40"}
            accepted = client.get(url, params=params)
            assert accepted.status_code == 202
            retry = client.post(f"{url}/retry", params=params)
            assert retry.status_code == 409
            assert retry.json()["error"]["type"] == "conflict"
            assert _count_jobs(app.state.store) == 1
            assert poll_job(client, accepted.json()["job_id"])["status"] == "done"


def test_restart_recovery(tmp_path: Path, bundle: BundleFixture) -> None:
    """Stale jobs are marked restarted and missing eager derivations are re-queued."""
    settings = _settings(tmp_path)
    store = Store(settings)
    result = ingest_staged(
        store, store.stage_from_directory(bundle.root, settings.extract_limits), name="x"
    )
    digest = result.digest
    running_id, queued_id = uuid.uuid4().hex, uuid.uuid4().hex
    _insert_job(
        store,
        job_id=running_id,
        status="running",
        kind="eager",
        digest=digest,
        member="dataset",
        deriver="overview",
        params_key="noparams",
    )
    _insert_job(
        store,
        job_id=queued_id,
        status="queued",
        kind="lazy",
        digest=digest,
        member="dataset",
        deriver="tenum",
        params_key="mode=a",
    )
    app = create_app(settings, static_dir=tmp_path / "nostatic")
    with viewer_client(app) as client:
        for job_id in (running_id, queued_id):
            info = client.get(f"/api/jobs/{job_id}").json()
            assert info["status"] == "failed"
            assert info["reason"] == "restart"
        latest = app.state.jobs.latest(digest, "dataset", "overview", "noparams")
        assert latest is not None and latest.kind == "eager"
        assert latest.job_id not in (running_id, queued_id)
        deadline = time.monotonic() + 60
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = client.get(f"/api/bundles/{digest}/status").json()
            if status["complete"]:
                break
            time.sleep(0.1)
        assert status["complete"] is True
        assert status["eager"]["done"] == 1
        overview = client.get(f"/api/bundles/{digest}/members/dataset/derived/overview")
        assert overview.status_code == 200
    before = _count_jobs(store)
    second = create_app(settings, static_dir=tmp_path / "nostatic")
    with viewer_client(second):
        assert _count_jobs(store) == before


def test_eager_after_upload_and_status(tmp_path: Path, zip_bytes: bytes) -> None:
    """Uploads start eager jobs and a failing lazy job shows up in the status failures."""
    app = make_app(tmp_path)
    with registered(Boom()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            deadline = time.monotonic() + 60
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                status = client.get(f"/api/bundles/{digest}/status").json()
                if status["complete"]:
                    break
                time.sleep(0.1)
            assert status["complete"] is True
            assert status["eager"]["done"] == 1
            boom = client.get(f"/api/bundles/{digest}/members/dataset/derived/tboom")
            assert boom.status_code == 202
            assert poll_job(client, boom.json()["job_id"])["status"] == "failed"
            status = client.get(f"/api/bundles/{digest}/status").json()
            failures = [item for item in status["failures"] if item["deriver"] == "tboom"]
            assert len(failures) == 1
            assert failures[0]["reason"] == "error"
            assert failures[0]["kind"] == "lazy"


def test_jobs_404(tmp_path: Path) -> None:
    """Unknown or malformed job ids and unknown bundles answer 404."""
    app = make_app(tmp_path)
    with viewer_client(app) as client:
        assert client.get(f"/api/jobs/{'0' * 32}").status_code == 404
        assert client.get("/api/jobs/nothex").status_code == 404
        assert client.get(f"/api/bundles/{'0' * 64}/status").status_code == 404


def test_delete_cancels_jobs(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a bundle cancels its running derivation and removes the files."""
    marks = _marks(tmp_path, monkeypatch)
    app = make_app(tmp_path)
    with registered(SlowDeriver()):
        with viewer_client(app) as client:
            digest = upload(client, zip_bytes).json()["digest"]
            url = f"/api/bundles/{digest}/members/dataset/derived/tslow"
            response = client.get(url, params={"ms": "60000", "tag": "30"})
            assert response.status_code == 202
            mark = _wait_for_mark(marks, "tslow-30")
            start = time.monotonic()
            deleted = client.request("DELETE", f"/api/bundles/{digest}", json={"confirm": digest})
            elapsed = time.monotonic() - start
            assert deleted.status_code == 200
            assert elapsed < 10
            _assert_pid_gone(int(mark["pid"]))
            assert not (app.state.store.bundles_dir / digest).exists()


def test_jobs_unit(tmp_path: Path, bundle: BundleFixture) -> None:
    """Directly exercise prepare, JobInfo parsing, invalid kinds and shutdown."""
    settings = _settings(tmp_path)
    store = Store(settings)
    result = ingest_staged(
        store, store.stage_from_directory(bundle.root, settings.extract_limits), name="x"
    )
    digest = result.digest
    with registered(SlowDeriver(), EnumDeriver()):
        with pytest.raises(DeriveNotFound):
            prepare(store, digest, "dataset", "nope", {})
        with pytest.raises(DeriveNotFound):
            prepare(store, digest, "scene", "overview", {})
        with pytest.raises(BadParams):
            prepare(store, digest, "dataset", "tenum", {"mode": "zzz"})
        prepared = prepare(store, digest, "dataset", "tenum", {"mode": "a"})
        manager = JobManager(store)
        try:
            with pytest.raises(ValueError):
                manager.submit(prepared, kind="bogus")
            job_id = uuid.uuid4().hex
            _insert_job(
                store,
                job_id=job_id,
                status="failed",
                kind="lazy",
                digest=digest,
                member="dataset",
                deriver="tenum",
                params_key="mode=a",
                error="timeout: x",
            )
            info = manager.get(job_id)
            assert info is not None
            assert info.reason == "timeout"
            assert info.error == "x"
        finally:
            manager.shutdown()
        with pytest.raises(RuntimeError):
            manager.submit(prepared)
