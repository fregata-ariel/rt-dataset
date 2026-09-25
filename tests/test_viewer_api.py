"""Tests for the viewer HTTP API (plateau_rt.viewer.api)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from viewer_bundle_fixtures import (
    BROKEN_CASE_MESSAGES,
    BROKEN_CASES,
    MALICIOUS_CASES,
    BundleFixture,
    make_archive,
    make_malicious_archive,
    write_broken_dataset,
    write_fixture_bundle,
)
from viewer_job_derivers import Boom, Versioned

from plateau_rt.application.rf_dataset_manifest import ManifestError
from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.api import create_app
from plateau_rt.viewer.derive import (
    register,
    registered,
    unregister,
)
from plateau_rt.viewer.derive.overview import OVERVIEW
from plateau_rt.viewer.ingest import IngestResult, ingest_staged
from plateau_rt.viewer.kinds import (
    BundleValidationError,
    dataset_manifest_from_bytes,
)
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store, compute_digest
from plateau_rt.viewer.testing import derive_in_temp_store

with warnings.catch_warnings():
    # Starlette 1.x warns that its TestClient will move from httpx to httpx2.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

MiB = 1 << 20
GOLDEN_PATH = Path(__file__).parent / "viewer_golden" / "derivers.json"
OCTET = {"Content-Type": "application/octet-stream"}


@pytest.fixture(scope="module")
def v3_bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    """Build the canonical v3 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("api-v3") / "bundle"
    return write_fixture_bundle(root, schema_version=3)


@pytest.fixture(scope="module")
def zip_bytes(v3_bundle: BundleFixture, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build the v3 fixture zip once per module and return its bytes."""
    out = tmp_path_factory.mktemp("api-zip") / "bundle.zip"
    make_archive(v3_bundle.root, out, "zip")
    return out.read_bytes()


@pytest.fixture(scope="module")
def tar_gz_bytes(v3_bundle: BundleFixture, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build a wrapped tar.gz of the v3 tree once per module and return its bytes."""
    out = tmp_path_factory.mktemp("api-targz") / "wrapped.tar.gz"
    make_archive(v3_bundle.root, out, "tar.gz", root_name="wrapped")
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
    """Build an app with no static directory and the eager hook disabled."""
    app = create_app(_settings(tmp_path, **overrides), static_dir=tmp_path / "nostatic")
    app.state.on_bundle_committed = lambda store, digest: None
    return app


def make_client(tmp_path: Path, **overrides: Any) -> tuple[Any, TestClient]:
    """Build an app and a TestClient for one test."""
    app = make_app(tmp_path, **overrides)
    return app, TestClient(app)


def upload(client: TestClient, data: bytes, name: str = "fixture") -> Any:
    """PUT an archive body as an octet-stream upload."""
    return client.put(f"/api/bundles/upload?name={name}", content=data, headers=OCTET)


def poll_job(client: TestClient, job_id: str, timeout: float = 60) -> dict[str, Any]:
    """Poll one derivation job until it is terminal and return its JSON body."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout} s")


def derive_ready(client: TestClient, url: str, params: Any = None) -> dict[str, Any]:
    """GET a derivation, polling its job on 202, and return the ready 200 body."""
    response = client.get(url, params=params)
    if response.status_code == 202:
        info = poll_job(client, response.json()["job_id"])
        assert info["status"] == "done", info
        response = client.get(url, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def assert_clean(store: Store) -> None:
    """Assert that a failed upload left no staging, index or bundle leftovers."""
    assert os.listdir(store.staging_dir) == []
    assert store.list() == []
    assert os.listdir(store.bundles_dir) == []


def _drive_asgi(
    app: Any,
    method: str,
    path: str,
    headers: Sequence[tuple[str, str]],
    chunks: Sequence[bytes],
    on_receive: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Drive the ASGI app and return the sent messages and the consumed chunk count."""
    consumed: list[int] = []
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        index = len(consumed)
        if index < len(chunks):
            if on_receive is not None:
                on_receive(index)
            consumed.append(index)
            return {
                "type": "http.request",
                "body": chunks[index],
                "more_body": index + 1 < len(chunks),
            }
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    asyncio.run(asyncio.wait_for(app(scope, receive, send), 30))
    return sent, len(consumed)


def _asgi_body(sent: Sequence[dict[str, Any]]) -> bytes:
    """Join the body chunks of an ASGI response."""
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def call_asgi(
    app: Any,
    method: str,
    path: str,
    headers: Sequence[tuple[str, str]],
    chunks: Sequence[bytes],
    on_receive: Callable[[int], None] | None = None,
) -> tuple[int, bytes, int]:
    """Drive the ASGI app with the body split into chunks; return (status, body, consumed)."""
    sent, consumed = _drive_asgi(app, method, path, headers, chunks, on_receive)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], _asgi_body(sent), consumed


def call_asgi_response(
    app: Any, method: str, path: str, headers: Sequence[tuple[str, str]] = ()
) -> tuple[int, dict[str, str], bytes]:
    """Drive the ASGI app without a body; return (status, response headers, body)."""
    sent, _consumed = _drive_asgi(app, method, path, headers, [])
    start = next(m for m in sent if m["type"] == "http.response.start")
    response_headers = {key.decode(): value.decode() for key, value in start.get("headers", [])}
    return start["status"], response_headers, _asgi_body(sent)


def test_upload_list_and_overview(
    tmp_path: Path, v3_bundle: BundleFixture, zip_bytes: bytes
) -> None:
    """AC1: upload, list, detail, derive overview, serve it and cache the second call."""
    _app, client = make_client(tmp_path)
    response = client.put(
        "/api/bundles/upload?name=fixture",
        content=zip_bytes,
        headers=OCTET,
    )
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"digest", "created", "members"}
    assert body["created"] is True
    digest = body["digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    assert digest == compute_digest(v3_bundle.root)[0]
    assert [member["id"] for member in body["members"]] == ["dataset", "scene", "p0", "p1"]
    assert [member["kind"] for member in body["members"]] == [
        "rf_dataset",
        "scene",
        "rf_partial",
        "rf_partial",
    ]
    dataset = next(member for member in body["members"] if member["id"] == "dataset")
    assert dataset["validation"]["summary"] == {
        "num_views": len(v3_bundle.truth.view_ids),
        "num_bs": len(v3_bundle.truth.bs_ids),
        "num_frequency_bins": len(v3_bundle.truth.frequency_offsets_hz),
    }

    listing = client.get("/api/bundles").json()
    assert [entry["digest"] for entry in listing["bundles"]] == [digest]
    assert listing["bundles"][0]["name"] == "fixture"
    detail = client.get(f"/api/bundles/{digest}").json()
    assert [member["id"] for member in detail["members"]] == ["dataset", "scene", "p0", "p1"]
    assert detail["validated_with"] == VIEWER_VERSION
    assert detail["error"] is None
    assert detail["derived"] == []

    accepted = client.get(f"/api/bundles/{digest}/members/dataset/derived/overview")
    assert accepted.status_code == 202
    job = poll_job(client, accepted.json()["job_id"])
    assert job["status"] == "done"
    info_body = client.get(f"/api/bundles/{digest}/members/dataset/derived/overview").json()
    assert info_body["cached"] is True
    assert {key: value for key, value in job["result"].items() if key != "cached"} == {
        key: value for key, value in info_body.items() if key != "cached"
    }
    assert info_body["deriver"] == "overview"
    assert info_body["member"] == "dataset"
    assert info_body["version"] == 1
    assert info_body["params_key"] == "noparams"
    assert [item["name"] for item in info_body["files"]] == ["overview.json"]
    file_url = info_body["files"][0]["url"]
    file_response = client.get(file_url)
    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith("application/json")
    served = file_response.content
    _, expected = derive_in_temp_store(OVERVIEW, v3_bundle.root, {}, member="dataset")
    assert served == expected["overview.json"]
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(served).hexdigest()
        == golden["overview"]["cases"]["v3/dataset/noparams"]["overview.json"]
    )
    parsed = json.loads(served)
    assert parsed["num_views"] == len(v3_bundle.truth.view_ids)
    assert parsed["num_bs"] == len(v3_bundle.truth.bs_ids)

    detail = client.get(f"/api/bundles/{digest}").json()
    assert len(detail["derived"]) == 1
    row = detail["derived"][0]
    assert (
        row["member"],
        row["deriver"],
        row["version"],
        row["params_key"],
        row["status"],
    ) == ("dataset", "overview", 1, "noparams", "ready")

    info2 = client.get(f"/api/bundles/{digest}/members/dataset/derived/overview").json()
    assert info2["cached"] is True
    assert info2["files"] == info_body["files"]
    _app.state.jobs.shutdown()


def test_hook_called_once_for_new_bundle(tmp_path: Path, zip_bytes: bytes) -> None:
    """The on_bundle_committed hook fires once per created bundle only."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    calls: list[tuple[Store, str]] = []
    app.state.on_bundle_committed = lambda s, d: calls.append((s, d))
    first = upload(client, zip_bytes, "first")
    assert first.status_code == 201
    assert calls == [(store, first.json()["digest"])]
    second = upload(client, zip_bytes, "second")
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert calls == [(store, first.json()["digest"])]
    failed = upload(client, b"hello", "bad")
    assert failed.status_code == 400
    assert calls == [(store, first.json()["digest"])]


def test_duplicate_upload_and_wrapped_targz(
    tmp_path: Path, zip_bytes: bytes, tar_gz_bytes: bytes
) -> None:
    """AC2: repeated and wrapped uploads reuse the same digest with created False."""
    app, client = make_client(tmp_path)
    first = upload(client, zip_bytes, "fixture")
    second = upload(client, zip_bytes, "again")
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["digest"] == first.json()["digest"]
    assert len(client.get("/api/bundles").json()["bundles"]) == 1
    third = upload(client, tar_gz_bytes, "wrapped")
    assert third.status_code == 200
    assert third.json()["created"] is False
    assert third.json()["digest"] == first.json()["digest"]


@pytest.mark.parametrize("case", BROKEN_CASES)
def test_broken_manifest(tmp_path: Path, case: str) -> None:
    """AC3: every broken manifest maps to a 400 validation error with the verbatim text."""
    bundle_root = tmp_path / "b"
    write_broken_dataset(bundle_root, case)
    archive = tmp_path / "broken.zip"
    make_archive(bundle_root, archive, "zip")
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    response = upload(client, archive.read_bytes(), "broken")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "validation"
    assert error["member"] == "dataset"
    manifest_bytes = (bundle_root / "dataset_manifest.json").read_bytes()
    try:
        dataset_manifest_from_bytes(manifest_bytes, ".")
    except ManifestError as exc:
        expected = str(exc)
    else:  # pragma: no cover - every broken case must raise
        raise AssertionError(f"broken case {case!r} did not raise ManifestError")
    assert error["message"] == expected
    assert BROKEN_CASE_MESSAGES[case] in error["message"]
    assert_clean(store)


def test_validation_error_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zip_bytes: bytes
) -> None:
    """A BundleValidationError reaches the client verbatim, newlines and HTML included."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store

    def fake_validate(raw_root: Path, *, max_array_bytes: int) -> list[Any]:
        raise BundleValidationError("dataset", "line one\nline <b>two</b>")

    monkeypatch.setattr("plateau_rt.viewer.ingest.validate_bundle", fake_validate)
    response = upload(client, zip_bytes, "verbatim")
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "type": "validation",
            "member": "dataset",
            "message": "line one\nline <b>two</b>",
        }
    }
    assert_clean(store)


def test_upload_unknown_kind(tmp_path: Path) -> None:
    """AC3: an archive without any known member maps to unknown_kind."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "ok.txt").write_bytes(b"ok\n")
    archive = tmp_path / "ok.zip"
    make_archive(source, archive, "zip")
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    response = upload(client, archive.read_bytes(), "unknown")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unknown_kind"
    assert_clean(store)


def _malicious_params() -> list[tuple[str, str]]:
    """Return the (case, format) pairs required by AC4."""
    pairs = [(case, "tar.gz") for case in MALICIOUS_CASES]
    pairs.extend((case, "zip") for case in ("dotdot", "symlink", "duplicate"))
    return pairs


@pytest.mark.parametrize(("case", "fmt"), _malicious_params())
def test_malicious_archive_rejected(tmp_path: Path, case: str, fmt: str) -> None:
    """AC4: every malicious archive is rejected with unsafe_archive and no leftovers."""
    archive = tmp_path / f"m_{case}.{fmt.replace('.', '_')}"
    make_malicious_archive(archive, case, fmt=fmt, file_count=1100)
    app, client = make_client(tmp_path, max_files=1000, max_extracted_bytes=4 * MiB)
    store: Store = app.state.store
    response = upload(client, archive.read_bytes(), "malicious")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "unsafe_archive"
    assert error["member"] is None
    assert_clean(store)


def test_non_archive_body(tmp_path: Path) -> None:
    """A non-archive body maps to unsafe_archive and leaves nothing behind."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    response = upload(client, b"hello", "nonarchive")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsafe_archive"
    assert_clean(store)


def test_content_length_over_limit(tmp_path: Path) -> None:
    """AC4: Content-Length over the cap answers 413 without reading the body."""
    app = make_app(tmp_path, max_upload_bytes=1000)
    store: Store = app.state.store
    status, body, consumed = call_asgi(
        app,
        "PUT",
        "/api/bundles/upload",
        [("content-type", "application/octet-stream"), ("content-length", "5000")],
        [b"x" * 500] * 10,
    )
    assert status == 413
    assert consumed == 0
    assert json.loads(body)["error"]["type"] == "too_large"
    assert_clean(store)


def test_chunked_over_limit(tmp_path: Path) -> None:
    """AC4: a chunked body over the cap stops right after the crossing chunk."""
    app = make_app(tmp_path, max_upload_bytes=1000)
    store: Store = app.state.store
    status, body, consumed = call_asgi(
        app,
        "PUT",
        "/api/bundles/upload",
        [("content-type", "application/octet-stream"), ("transfer-encoding", "chunked")],
        [b"x" * 300] * 10,
    )
    assert status == 413
    assert consumed == 4
    assert json.loads(body)["error"]["type"] == "too_large"
    assert_clean(store)


def test_insufficient_storage(tmp_path: Path) -> None:
    """AC4: a free-space cap above the disk free bytes answers 507 before reading."""
    app = make_app(tmp_path, max_extracted_bytes=1 << 60)
    store: Store = app.state.store
    status, body, consumed = call_asgi(
        app,
        "PUT",
        "/api/bundles/upload",
        [("content-type", "application/octet-stream")],
        [b"x" * 10],
    )
    assert status == 507
    assert consumed == 0
    assert json.loads(body)["error"]["type"] == "insufficient_storage"
    assert_clean(store)


def test_bad_params_before_reading(tmp_path: Path) -> None:
    """AC4: a bad Content-Type or name answers bad_params and leaves nothing."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    response = client.put(
        "/api/bundles/upload",
        content=b"x",
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "bad_params"
    assert_clean(store)
    response = client.put(
        "/api/bundles/upload?name=a%0Ab",
        content=b"x",
        headers=OCTET,
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "bad_params"
    assert_clean(store)


def test_upload_streams_into_staging_only(
    tmp_path: Path, zip_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC5: the uploaded body is streamed into one staging file and nothing else."""
    sys_tmp = tmp_path / "systmp"
    sys_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))
    monkeypatch.setenv("TMPDIR", str(sys_tmp))
    app = make_app(tmp_path)
    store: Store = app.state.store
    chunks = [zip_bytes[i : i + 65536] for i in range(0, len(zip_bytes), 65536)]
    recorded: dict[str, Any] = {}

    def on_receive(index: int) -> None:
        if index == 1:
            entries = sorted(store.staging_dir.iterdir())
            recorded["names"] = [entry.name for entry in entries]
            recorded["sizes"] = [entry.stat().st_size for entry in entries]

    status, _body, _consumed = call_asgi(
        app,
        "PUT",
        "/api/bundles/upload",
        [("content-type", "application/octet-stream"), ("content-length", str(len(zip_bytes)))],
        chunks,
        on_receive=on_receive,
    )
    assert status == 201
    assert len(recorded["names"]) == 1
    assert recorded["names"][0].endswith(".upload")
    assert recorded["sizes"] == [len(chunks[0])]
    assert os.listdir(sys_tmp) == []
    assert os.listdir(store.staging_dir) == []


def _raw_bundle(tmp_path: Path, v3_bundle: BundleFixture) -> tuple[Any, TestClient, str]:
    """Copy the v3 tree with HTML/SVG, upload it and add a secret outside the raw tree."""
    source = tmp_path / "bundle"
    shutil.copytree(v3_bundle.root, source)
    (source / "dataset" / "notes.html").write_bytes(b"<script>alert(1)</script>")
    (source / "dataset" / "pic.svg").write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    )
    archive = tmp_path / "raw.zip"
    make_archive(source, archive, "zip")
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    response = upload(client, archive.read_bytes(), "raw")
    assert response.status_code == 201
    (store.data_dir / "secret.txt").write_bytes(b"secret")
    return app, client, response.json()["digest"]


def test_raw_serving_headers_and_bytes(tmp_path: Path, v3_bundle: BundleFixture) -> None:
    """AC6: raw files are served as octet-streams with the four security headers."""
    _app, client, digest = _raw_bundle(tmp_path, v3_bundle)
    cases = {
        "members/dataset/raw/notes.html": b"<script>alert(1)</script>",
        "members/dataset/raw/pic.svg": b'<svg xmlns="http://www.w3.org/2000/svg">'
        b"<script>alert(1)</script></svg>",
    }
    for suffix, expected in cases.items():
        response = client.get(f"/api/bundles/{digest}/{suffix}")
        assert response.status_code == 200
        assert response.content == expected
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-disposition"].startswith("attachment")
        assert response.headers["content-security-policy"] == "sandbox"
        assert response.headers["cache-control"] == "private, max-age=0"
    manifest = client.get(f"/api/bundles/{digest}/members/dataset/raw/dataset_manifest.json")
    assert manifest.status_code == 200
    assert manifest.headers["content-type"] == "application/octet-stream"
    scene = client.get(f"/api/bundles/{digest}/members/scene/raw/box.ply")
    assert scene.status_code == 200


def test_raw_traversal_is_blocked(tmp_path: Path, v3_bundle: BundleFixture) -> None:
    """AC6: traversal and member escapes return the envelope and leak nothing."""
    app, client, digest = _raw_bundle(tmp_path, v3_bundle)
    store: Store = app.state.store
    manifest_bytes = (v3_bundle.root / "dataset" / "dataset_manifest.json").read_bytes()
    base = f"/api/bundles/{digest}/members/dataset"
    raw_paths = [
        "raw/../../../secret.txt",
        "raw/%2e%2e/%2e%2e/%2e%2e/secret.txt",
        "raw/..%2F..%2F..%2Fsecret.txt",
        "raw/%2Fetc%2Fpasswd",
        "raw/..%5c..%5csecret.txt",
        "raw/%2e%2e/%2e%2e/%2e%2e/index.sqlite",
    ]
    for raw_path in raw_paths:
        response = client.get(f"{base}/{raw_path}")
        assert response.status_code in (400, 404)
        body = response.json()
        assert set(body) == {"error"}
        assert body["error"]["type"] in ("bad_params", "not_found")
        for needle in (b"secret", b"SQLite format 3", manifest_bytes):
            assert needle not in response.content
    member_escape = client.get(
        f"/api/bundles/{digest}/members/scene/raw/..%2Fdataset%2Fdataset_manifest.json"
    )
    assert member_escape.status_code in (400, 404)
    assert b"SQLite format 3" not in member_escape.content
    assert (store.data_dir / "secret.txt").read_bytes() == b"secret"
    for raw_path in (
        "raw/%2e%2e/%2e%2e/%2e%2e/secret.txt",
        "raw/..%2F..%2F..%2Fsecret.txt",
        "raw/%2e%2e/%2e%2e/%2e%2e/index.sqlite",
    ):
        response = client.get(f"{base}/{raw_path}")
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "bad_params"


def test_raw_missing_and_unknown(tmp_path: Path, v3_bundle: BundleFixture) -> None:
    """AC6: directories, missing files and unknown ids all answer 404 not_found."""
    _app, client, digest = _raw_bundle(tmp_path, v3_bundle)
    assert client.get(f"/api/bundles/{digest}/members/dataset/raw/views").status_code == 404
    assert client.get(f"/api/bundles/{digest}/members/dataset/raw/nope.txt").status_code == 404
    assert client.get(f"/api/bundles/{digest}/members/nomember/raw/x").status_code == 404
    assert client.get(f"/api/bundles/{'0' * 64}/members/dataset/raw/x").status_code == 404
    assert client.get("/api/bundles/notadigest/members/dataset/raw/x").status_code == 404


def test_derived_file_etag(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC7: derived bytes, ETag and If-None-Match handling."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    digest = upload(client, zip_bytes).json()["digest"]
    info = derive_ready(client, f"/api/bundles/{digest}/members/dataset/derived/overview")
    url = info["files"][0]["url"]
    response = client.get(url)
    assert response.status_code == 200
    data = (
        store.derived_dir(digest)
        / "dataset"
        / "overview"
        / "v1"
        / "noparams"
        / "nolink"
        / "overview.json"
    ).read_bytes()
    assert response.content == data
    etag = response.headers["etag"]
    assert etag == '"' + hashlib.sha256(data).hexdigest() + '"'
    assert etag == '"' + info["files"][0]["sha256"] + '"'
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    not_modified = client.get(url, headers={"If-None-Match": etag})
    assert not_modified.status_code == 304
    assert not_modified.content == b""
    assert not_modified.headers["etag"] == etag
    assert client.get(url, headers={"If-None-Match": "W/" + etag}).status_code == 304
    assert client.get(url, headers={"If-None-Match": f'"other", {etag}'}).status_code == 304
    assert client.get(url, headers={"If-None-Match": '"other"'}).status_code == 200
    assert client.get(url, headers={"If-None-Match": "*"}).status_code == 304
    app.state.jobs.shutdown()


def test_derived_version_bump(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC7: versioned URLs change with the deriver version and old versions 404."""
    app, client = make_client(tmp_path)
    store: Store = app.state.store
    digest = upload(client, zip_bytes).json()["digest"]
    base = f"/api/bundles/{digest}/members/dataset/derived/tversion"
    with registered(Versioned(1)):
        info = derive_ready(client, base, {"mode": "a/b"})
        assert info["params_key"] == "mode=a%2Fb"
        assert info["links_key"] == "nolink"
        for item in info["files"]:
            assert "/v1/" in item["url"]
            assert "mode=a%252Fb" in item["url"]
        for item in info["files"]:
            status, headers, body = call_asgi_response(
                app, "GET", urllib.parse.unquote(item["url"])
            )
            assert status == 200
            data = (
                store.derived_dir(digest)
                / "dataset"
                / "tversion"
                / "v1"
                / "mode=a%2Fb"
                / "nolink"
                / item["name"]
            ).read_bytes()
            assert body == data
            if item["name"] == "blob.bin":
                assert headers["content-type"] == "application/octet-stream"
        v1_urls = [item["url"] for item in info["files"]]
    register(Versioned(2), replace=True)
    try:
        info2 = derive_ready(client, base, {"mode": "a/b"})
        assert all("/v2/" in item["url"] for item in info2["files"])
        assert [item["url"] for item in info2["files"]] != v1_urls
        for url in v1_urls:
            status, _headers, _body = call_asgi_response(app, "GET", urllib.parse.unquote(url))
            assert status == 404
        out = next(item for item in info2["files"] if item["name"] == "out.json")
        status, _headers, body = call_asgi_response(app, "GET", urllib.parse.unquote(out["url"]))
        assert status == 200
        assert json.loads(body)["version"] == 2
    finally:
        unregister("tversion")
        app.state.jobs.shutdown()


def test_derived_file_404s(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC7: meta, unknown names, odd keys and unknown derivers answer 404."""
    app, client, digest = _upload_and_digest(tmp_path, zip_bytes)
    info = derive_ready(client, f"/api/bundles/{digest}/members/dataset/derived/overview")
    url = info["files"][0]["url"]
    directory_url = url.rsplit("/", 1)[0]
    assert client.get(f"{directory_url}/_meta.json").status_code == 404
    assert client.get(f"{directory_url}/nope.bin").status_code == 404
    odd = f"/api/bundles/{digest}/members/dataset/derived/overview/v1/%2e%2e/nolink/overview.json"
    assert client.get(odd).status_code == 404
    wrong_links = (
        f"/api/bundles/{digest}/members/dataset/derived/overview/v1/noparams/self/overview.json"
    )
    assert client.get(wrong_links).status_code == 404
    unknown = f"/api/bundles/{digest}/members/dataset/derived/nope/v1/noparams/nolink/x.json"
    assert client.get(unknown).status_code == 404
    app.state.jobs.shutdown()


def _upload_and_digest(tmp_path: Path, zip_bytes: bytes) -> tuple[Any, TestClient, str]:
    """Upload the fixture zip and return (app, client, digest)."""
    app, client = make_client(tmp_path)
    response = upload(client, zip_bytes)
    assert response.status_code == 201
    return app, client, response.json()["digest"]


def test_derivers_listing(tmp_path: Path) -> None:
    """GET /api/derivers lists overview and a temporarily registered enum deriver."""
    _app, client = make_client(tmp_path)
    body = client.get("/api/derivers").json()
    overview = next(item for item in body["derivers"] if item["name"] == "overview")
    assert overview == {
        "name": "overview",
        "version": 1,
        "kinds": ["rf_dataset"],
        "eager": True,
        "params": [],
    }
    with registered(Versioned(1)):
        body = client.get("/api/derivers").json()
        tversion = next(item for item in body["derivers"] if item["name"] == "tversion")
        assert tversion["params"][0]["values"] == ["a/b", "c d"]


def test_derive_error_mapping(tmp_path: Path, zip_bytes: bytes) -> None:
    """Unknown params, repeated params, wrong kind and failures map to the error table."""
    _app, client, digest = _upload_and_digest(tmp_path, zip_bytes)
    base = f"/api/bundles/{digest}/members/dataset/derived"
    unknown_param = client.get(f"{base}/overview", params={"x": "1"})
    assert unknown_param.status_code == 400
    error = unknown_param.json()["error"]
    assert error["type"] == "bad_params"
    assert error["member"] == "dataset"
    with registered(Versioned(1)):
        assert client.get(f"{base}/tversion", params={"mode": "zzz"}).status_code == 400
        repeated = client.get(f"{base}/tversion", params=[("mode", "a/b"), ("mode", "c d")])
        assert repeated.status_code == 400
        assert repeated.json()["error"]["type"] == "bad_params"
    assert client.get(f"{base}/nope").status_code == 404
    scene = client.get(f"/api/bundles/{digest}/members/scene/derived/overview")
    assert scene.status_code == 404
    with registered(Boom()):
        accepted = client.get(f"{base}/tboom")
        assert accepted.status_code == 202
        info = poll_job(client, accepted.json()["job_id"])
        assert info["status"] == "failed"
        assert info["reason"] == "error"
        assert "boom" in info["error"]
    _app.state.jobs.shutdown()


def test_health(tmp_path: Path) -> None:
    """GET /api/health reports the viewer version and read-only flag."""
    _app, client = make_client(tmp_path)
    assert client.get("/api/health").json() == {
        "status": "ok",
        "viewer_version": VIEWER_VERSION,
        "read_only": False,
        "store_schema_version": 1,
    }
    app_ro, client_ro = make_client(tmp_path / "ro", read_only=True)
    assert client_ro.get("/api/health").json()["read_only"] is True
    assert app_ro.state.settings.read_only is True


def test_delete_bundle(tmp_path: Path, zip_bytes: bytes) -> None:
    """DELETE requires the matching digest and removes the bundle."""
    app, client, digest = _upload_and_digest(tmp_path, zip_bytes)
    store: Store = app.state.store
    url = f"/api/bundles/{digest}"
    assert client.request("DELETE", url, json={"confirm": "0" * 64}).status_code == 400
    assert client.get("/api/bundles").json()["bundles"]
    assert client.request("DELETE", url, json={}).status_code == 400
    assert client.request("DELETE", url).status_code == 400
    deleted = client.request("DELETE", url, json={"confirm": digest})
    assert deleted.status_code == 200
    assert deleted.json() == {"digest": digest, "deleted": True}
    assert client.get("/api/bundles").json()["bundles"] == []
    assert client.get(url).status_code == 404
    assert client.get(f"{url}/members/dataset/raw/dataset_manifest.json").status_code == 404
    assert client.request("DELETE", url, json={"confirm": digest}).status_code == 404
    assert os.listdir(store.staging_dir) == []


def test_static_hook(tmp_path: Path) -> None:
    """GET / serves index.html and /static is mounted when the directory exists."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_bytes(b"<h1>viewer</h1>")
    (static / "app.js").write_bytes(b"console.log(1)")
    app = create_app(_settings(tmp_path), static_dir=static)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.content == b"<h1>viewer</h1>"
    assert client.get("/static/app.js").status_code == 200


def test_no_static(tmp_path: Path) -> None:
    """Without a static directory GET / is plain text and /static is absent."""
    app = create_app(_settings(tmp_path), static_dir=tmp_path / "nostatic")
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "/api/health" in response.text
    assert "docs/viewer.md" in response.text
    assert client.get("/static/app.js").status_code == 404


def test_unknown_api_route(tmp_path: Path) -> None:
    """An unknown /api route answers the not_found envelope."""
    _app, client = make_client(tmp_path)
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"


def test_ingest_staged_without_http(tmp_path: Path, v3_bundle: BundleFixture) -> None:
    """ingest_staged commits a staged root, reuses a duplicate and cleans up a failure."""
    store = Store(_settings(tmp_path))
    limits = store.settings.extract_limits
    first = ingest_staged(store, store.stage_from_directory(v3_bundle.root, limits), name="a")
    assert isinstance(first, IngestResult)
    assert first.created is True
    assert first.digest == compute_digest(v3_bundle.root)[0]
    assert [member["id"] for member in first.members] == ["dataset", "scene", "p0", "p1"]
    again = ingest_staged(store, store.stage_from_directory(v3_bundle.root, limits), name="b")
    assert (again.digest, again.created) == (first.digest, False)
    assert again.members == first.members
    assert os.listdir(store.staging_dir) == []
    broken = tmp_path / "broken"
    write_broken_dataset(broken, "duplicate_view_id")
    staged = store.stage_from_directory(broken, limits)
    with pytest.raises(BundleValidationError) as excinfo:
        ingest_staged(store, staged, name="broken")
    assert excinfo.value.member_id == "dataset"
    assert BROKEN_CASE_MESSAGES["duplicate_view_id"] in excinfo.value.message
    assert os.listdir(store.staging_dir) == []
    assert [record.digest for record in store.list()] == [first.digest]
