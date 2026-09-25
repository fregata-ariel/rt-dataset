"""Tests for the viewer web security baseline (headers, CSRF and the Host allow-list)."""

from __future__ import annotations

import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import pytest
from viewer_bundle_fixtures import BundleFixture, make_archive, write_fixture_bundle
from viewer_client import CLIENT_BASE_URL, viewer_client

from plateau_rt.viewer.api.app import DEFAULT_STATIC_DIR, create_app
from plateau_rt.viewer.api.security import (
    CONTENT_SECURITY_POLICY,
    CSRF_HEADER,
    csrf_violation,
    merge_csp,
    origin_allowed,
)
from plateau_rt.viewer.settings import ViewerSettings

with warnings.catch_warnings():
    # Starlette 1.x warns that its TestClient will move from httpx to httpx2.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

MiB = 1 << 20
OCTET = {"Content-Type": "application/octet-stream"}
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "innerHTML": re.compile(r"\binnerHTML\b"),
    "outerHTML": re.compile(r"\bouterHTML\b"),
    "insertAdjacentHTML": re.compile(r"\binsertAdjacentHTML\b"),
    "document.write": re.compile(r"document\s*\.\s*write(ln)?\b"),
    "eval": re.compile(r"\beval\s*\("),
    "new Function": re.compile(r"\bnew\s+Function\b"),
    "createContextualFragment": re.compile(r"\bcreateContextualFragment\b"),
}


@pytest.fixture(scope="module")
def v3_bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    """Build the canonical v3 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("security-v3") / "bundle"
    return write_fixture_bundle(root, schema_version=3)


@pytest.fixture(scope="module")
def zip_bytes(v3_bundle: BundleFixture, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Build the v3 fixture zip once per module and return its bytes."""
    out = tmp_path_factory.mktemp("security-zip") / "bundle.zip"
    make_archive(v3_bundle.root, out, "zip")
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


def make_app(tmp_path: Path, *, static_dir: Path | None = None, **overrides: Any) -> Any:
    """Build an app with the eager hook disabled and an optional static directory."""
    directory = tmp_path / "nostatic" if static_dir is None else static_dir
    app = create_app(_settings(tmp_path, **overrides), static_dir=directory)
    app.state.on_bundle_committed = lambda store, digest: None
    return app


def make_client(tmp_path: Path, **overrides: Any) -> tuple[Any, TestClient]:
    """Build an app and a CSRF-enabled TestClient for one test."""
    app = make_app(tmp_path, **overrides)
    return app, viewer_client(app)


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


def test_csrf_upload_without_header_rejected(
    tmp_path: Path, v3_bundle: BundleFixture, zip_bytes: bytes
) -> None:
    """AC1: an upload without the CSRF header is 403 and staging keeps no upload file."""
    app = make_app(tmp_path)
    store = app.state.store
    raw = TestClient(app, base_url=CLIENT_BASE_URL)
    response = raw.put("/api/bundles/upload?name=csrf", content=zip_bytes, headers=OCTET)
    assert response.status_code == 403
    error = response.json()["error"]
    assert error["type"] == "forbidden"
    assert error["member"] is None
    assert CSRF_HEADER in error["message"]
    assert raw.get("/api/bundles").json()["bundles"] == []
    assert not list(store.staging_dir.glob("*.upload"))


def test_csrf_retry_and_delete_without_header(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC1: retry and delete without the CSRF header are 403 and leave the bundle intact."""
    app, client = make_client(tmp_path)
    digest = upload(client, zip_bytes).json()["digest"]
    raw = TestClient(app, base_url=CLIENT_BASE_URL)
    retry = raw.post(f"/api/bundles/{digest}/members/dataset/derived/overview/retry")
    assert retry.status_code == 403
    assert retry.json()["error"]["type"] == "forbidden"
    deleted = raw.request("DELETE", f"/api/bundles/{digest}", json={"confirm": digest})
    assert deleted.status_code == 403
    assert deleted.json()["error"]["type"] == "forbidden"
    assert raw.get(f"/api/bundles/{digest}").status_code == 200
    app.state.jobs.shutdown()


@pytest.mark.parametrize("value", ["0", "true", "11"])
def test_csrf_wrong_header_value(tmp_path: Path, value: str) -> None:
    """AC1: a present but wrong X-Viewer-Request value is 403."""
    app, _client = make_client(tmp_path)
    raw = TestClient(app, base_url=CLIENT_BASE_URL, headers={CSRF_HEADER: value})
    response = raw.put("/api/bundles/upload?name=x", content=b"x", headers=OCTET)
    assert response.status_code == 403
    assert CSRF_HEADER in response.json()["error"]["message"]


@pytest.mark.parametrize("method", ["PUT", "POST", "DELETE"])
@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://127.0.0.1:9999"])
def test_csrf_foreign_origin_rejected(
    tmp_path: Path, zip_bytes: bytes, method: str, origin: str
) -> None:
    """AC1: an unsafe request with a foreign Origin is 403 even with the CSRF header."""
    app, client = make_client(tmp_path)
    digest = upload(client, zip_bytes).json()["digest"]
    raw = TestClient(app, base_url=CLIENT_BASE_URL, headers={CSRF_HEADER: "1", "Origin": origin})
    if method == "PUT":
        response = raw.put("/api/bundles/upload?name=x", content=zip_bytes, headers=OCTET)
    elif method == "POST":
        response = raw.post(f"/api/bundles/{digest}/members/dataset/derived/overview/retry")
    else:
        response = raw.request("DELETE", f"/api/bundles/{digest}", json={"confirm": digest})
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "forbidden"
    assert origin in response.json()["error"]["message"]
    app.state.jobs.shutdown()


def test_csrf_same_origin_passes(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC1: a same-origin unsafe request passes with the CSRF header."""
    app, _client = make_client(tmp_path)
    raw = TestClient(
        app, base_url=CLIENT_BASE_URL, headers={CSRF_HEADER: "1", "Origin": "http://127.0.0.1"}
    )
    response = raw.put("/api/bundles/upload?name=ok", content=zip_bytes, headers=OCTET)
    assert response.status_code == 201


def test_csrf_allowed_origin_passes_and_is_opt_in(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC1: an allow-listed Origin passes, the same Origin is rejected when the list is empty."""
    allowed_app, _client = make_client(
        tmp_path / "allowed", allowed_origins=("https://viewer.example.org",)
    )
    allowed = TestClient(
        allowed_app,
        base_url=CLIENT_BASE_URL,
        headers={CSRF_HEADER: "1", "Origin": "https://viewer.example.org"},
    )
    allowed_upload = allowed.put("/api/bundles/upload?name=ok", content=zip_bytes, headers=OCTET)
    assert allowed_upload.status_code == 201

    blocked_app, _client = make_client(tmp_path / "blocked")
    blocked = TestClient(
        blocked_app,
        base_url=CLIENT_BASE_URL,
        headers={CSRF_HEADER: "1", "Origin": "https://viewer.example.org"},
    )
    blocked_response = blocked.put("/api/bundles/upload?name=ok", content=zip_bytes, headers=OCTET)
    assert blocked_response.status_code == 403


def test_safe_methods_are_never_csrf_blocked(tmp_path: Path) -> None:
    """AC1: GET, HEAD and OPTIONS never require the CSRF header, even from a foreign Origin."""
    app, _client = make_client(tmp_path)
    raw = TestClient(app, base_url=CLIENT_BASE_URL, headers={"Origin": "http://evil.example"})
    assert raw.get("/api/bundles").status_code == 200
    assert raw.head("/api/bundles").status_code != 403
    assert raw.request("OPTIONS", "/api/bundles").status_code != 403


def test_full_frontend_flow_with_csrf(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC1: the normal upload, derive and delete flow passes with header and same-origin Origin."""
    app, client = make_client(tmp_path)
    origin = {"Origin": "http://127.0.0.1"}
    created = client.put(
        "/api/bundles/upload?name=flow", content=zip_bytes, headers={**OCTET, **origin}
    )
    assert created.status_code == 201
    digest = created.json()["digest"]
    assert client.get(f"/api/bundles/{digest}", headers=origin).status_code == 200
    info = derive_ready(client, f"/api/bundles/{digest}/members/dataset/derived/overview")
    assert info["cached"] is True
    deleted = client.request(
        "DELETE", f"/api/bundles/{digest}", json={"confirm": digest}, headers=origin
    )
    assert deleted.status_code == 200
    app.state.jobs.shutdown()


@pytest.mark.parametrize(
    ("origin", "host", "allowed", "expected"),
    [
        ("http://127.0.0.1:8765", "127.0.0.1:8765", (), True),
        ("HTTP://127.0.0.1", "127.0.0.1", (), True),
        ("http://127.0.0.1/x", None, (), False),
        ("http://u@x", None, (), False),
        ("javascript:alert(1)", "x", (), False),
        ("", None, (), False),
        ("null", None, (), False),
        ("https://viewer.example.org", "127.0.0.1", ("https://viewer.example.org",), True),
        ("https://viewer.example.org", "127.0.0.1", (), False),
    ],
)
def test_origin_allowed(
    origin: str, host: str | None, allowed: tuple[str, ...], expected: bool
) -> None:
    """origin_allowed accepts only http(s) origins that match the host or the allow-list."""
    assert origin_allowed(origin, host, allowed) is expected


def test_csrf_violation_rules() -> None:
    """csrf_violation lets safe methods and valid headers through and flags the rest."""
    assert csrf_violation("GET", {}, ()) is None
    assert csrf_violation("HEAD", {}, ()) is None
    assert csrf_violation("OPTIONS", {}, ()) is None
    assert csrf_violation("PATCH", {}, ()) is not None
    assert CSRF_HEADER in (csrf_violation("PUT", {}, ()) or "")
    assert csrf_violation("PUT", {CSRF_HEADER: "1"}, ()) is None
    same_origin = {CSRF_HEADER: "1", "Origin": "http://127.0.0.1"}
    assert csrf_violation("PUT", {**same_origin, "Host": "127.0.0.1"}, ()) is None
    foreign = {CSRF_HEADER: "1", "Origin": "http://evil.example", "Host": "127.0.0.1"}
    assert csrf_violation("PUT", foreign, ()) is not None


@pytest.mark.parametrize(
    "host", ["evil.example", "127.0.0.1.evil.example", "localhost.evil.example"]
)
def test_host_rejected(tmp_path: Path, host: str) -> None:
    """AC1: a Host outside the allow-list answers 400."""
    app, client = make_client(tmp_path)
    assert client.get("/api/health", headers={"host": host}).status_code == 400


def test_bad_host_upload(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC1: an upload from a bad Host is 400 even with the CSRF header."""
    app, client = make_client(tmp_path)
    response = client.put(
        "/api/bundles/upload?name=x", content=zip_bytes, headers={**OCTET, "host": "evil.example"}
    )
    assert response.status_code == 400


@pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1:8000"])
def test_host_allowed_with_port(tmp_path: Path, host: str) -> None:
    """AC1: an allowed Host with any port passes the allow-list."""
    app, client = make_client(tmp_path)
    assert client.get("/api/health", headers={"host": host}).status_code == 200


def test_allowed_hosts_setting(tmp_path: Path) -> None:
    """AC1: VIEWER_ALLOWED_HOSTS replaces the default host list."""
    app, client = make_client(tmp_path, allowed_hosts=("viewer.example.org",))
    assert client.get("/api/health", headers={"host": "viewer.example.org"}).status_code == 200
    assert client.get("/api/health", headers={"host": "127.0.0.1"}).status_code == 400


def _assert_security_headers(response: Any) -> None:
    """Assert the three baseline security headers are present on ``response``."""
    assert response.headers["content-security-policy"].startswith(CONTENT_SECURITY_POLICY)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_security_headers_on_every_response(tmp_path: Path, zip_bytes: bytes) -> None:
    """AC2: success, redirect, error and raw-file responses all carry the baseline headers."""
    app = make_app(tmp_path, static_dir=DEFAULT_STATIC_DIR)
    client = viewer_client(app)
    responses: list[Any] = []
    index = client.get("/")
    assert index.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    responses.append(index)
    build = app.state.static_build
    asset = client.get(f"/static/{build}/js/app.js")
    responses.append(asset)
    assert asset.status_code == 200
    not_modified = client.get(
        f"/static/{build}/js/app.js", headers={"If-None-Match": asset.headers["etag"]}
    )
    responses.append(not_modified)
    responses.append(client.get("/api/health"))
    responses.append(client.get("/api/bundles"))
    responses.append(client.get("/api/does-not-exist"))
    responses.append(client.get("/api/health", headers={"host": "evil.example"}))
    raw_client = TestClient(app, base_url=CLIENT_BASE_URL)
    responses.append(raw_client.put("/api/bundles/upload?name=x", content=b"x", headers=OCTET))
    responses.append(client.put("/api/bundles/upload?name=x", content=b"garbage", headers=OCTET))
    digest = upload(client, zip_bytes, "raw").json()["digest"]
    raw_file = client.get(f"/api/bundles/{digest}/members/dataset/raw/dataset_manifest.json")
    assert raw_file.headers["content-security-policy"] == merge_csp("sandbox")
    assert raw_file.headers["content-security-policy"].endswith("; sandbox")
    responses.append(raw_file)
    for response in responses:
        _assert_security_headers(response)
    app.state.jobs.shutdown()


def test_security_headers_on_unhandled_500(tmp_path: Path) -> None:
    """AC2: an unhandled exception still answers 500 with the baseline headers."""
    app = make_app(tmp_path)

    @app.get("/api/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    client = viewer_client(app, raise_server_exceptions=False)
    response = client.get("/api/boom")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    _assert_security_headers(response)
    with pytest.raises(RuntimeError, match="boom"):
        viewer_client(app).get("/api/boom")
    app.state.jobs.shutdown()


def test_merge_csp_keeps_the_baseline() -> None:
    """merge_csp keeps the baseline and appends a non-empty route policy."""
    assert merge_csp(None) == CONTENT_SECURITY_POLICY
    assert merge_csp("") == CONTENT_SECURITY_POLICY
    assert merge_csp("   ") == CONTENT_SECURITY_POLICY
    assert merge_csp("sandbox") == CONTENT_SECURITY_POLICY + "; sandbox"
    assert merge_csp(" sandbox ") == CONTENT_SECURITY_POLICY + "; sandbox"


def scanned_files(static_dir: Path) -> list[str]:
    """Return the sorted ``.js``/``.html`` relpaths scanned under ``static_dir``."""
    files: list[str] = []
    for path in sorted(static_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".js", ".html"):
            continue
        rel = path.relative_to(static_dir)
        if rel.parts[0] == "vendor":
            continue
        files.append(rel.as_posix())
    return files


def scan_static(static_dir: Path) -> list[tuple[str, str]]:
    """Return ``(relpath, pattern_name)`` for every forbidden API under ``static_dir``."""
    hits: list[tuple[str, str]] = []
    for rel in scanned_files(static_dir):
        text = (static_dir / rel).read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                hits.append((rel, name))
    return hits


def test_static_scan_is_clean() -> None:
    """AC5: the real frontend never uses a forbidden markup sink."""
    assert scan_static(DEFAULT_STATIC_DIR) == []


def test_scanned_files_cover_the_frontend() -> None:
    """AC5: the scan walks the real DOM, app, API and panel sources plus index.html."""
    files = scanned_files(DEFAULT_STATIC_DIR)
    assert files
    for rel in ("js/dom.js", "js/app.js", "js/api.js", "js/panels/bundles.js", "index.html"):
        assert rel in files


def test_static_scan_detects_inner_html(tmp_path: Path) -> None:
    """AC5: adding one innerHTML assignment makes the scan fail."""
    copy = tmp_path / "static"
    shutil.copytree(DEFAULT_STATIC_DIR, copy)
    target = copy / "js" / "panels" / "bundles.js"
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\nel.innerHTML = name;\n")
    assert scan_static(copy) == [("js/panels/bundles.js", "innerHTML")]


@pytest.mark.parametrize(
    ("pattern_name", "snippet"),
    [
        ("innerHTML", "el.innerHTML = x;"),
        ("outerHTML", "el.outerHTML = x;"),
        ("insertAdjacentHTML", 'el.insertAdjacentHTML("beforeend", x);'),
        ("document.write", "document.write(x);"),
        ("document.write", "document . writeln(x);"),
        ("eval", "eval(x);"),
        ("new Function", 'new Function("x");'),
        ("createContextualFragment", "range.createContextualFragment(x);"),
    ],
)
def test_static_scan_detects_each_pattern(tmp_path: Path, pattern_name: str, snippet: str) -> None:
    """AC5: every forbidden pattern is detected in an otherwise clean copy."""
    copy = tmp_path / "static"
    shutil.copytree(DEFAULT_STATIC_DIR, copy)
    target = copy / "js" / "panels" / "bundles.js"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{snippet}\n")
    assert scan_static(copy) == [("js/panels/bundles.js", pattern_name)]


def test_static_scan_ignores_vendor(tmp_path: Path) -> None:
    """AC5: files under vendor/ are not scanned."""
    copy = tmp_path / "static"
    shutil.copytree(DEFAULT_STATIC_DIR, copy)
    (copy / "vendor" / "fake.js").write_text("el.innerHTML = x;\n", encoding="utf-8")
    assert scan_static(copy) == []


def plotly_without_escape(static_dir: Path) -> list[str]:
    """Return every panel file that references Plotly without importing escapeHtml."""
    offenders: list[str] = []
    for path in sorted((static_dir / "js" / "panels").rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        if "Plotly" in text and "escapeHtml" not in text:
            offenders.append(path.relative_to(static_dir).as_posix())
    return offenders


def test_panels_using_plotly_escape_html(tmp_path: Path) -> None:
    """AC5: every Plotly panel escapes its data strings."""
    assert plotly_without_escape(DEFAULT_STATIC_DIR) == []
    copy = tmp_path / "static"
    panels = copy / "js" / "panels"
    panels.mkdir(parents=True)
    (panels / "fake.js").write_text("Plotly.newPlot(el, [{text: name}]);\n", encoding="utf-8")
    assert plotly_without_escape(copy) == ["js/panels/fake.js"]


def test_frontend_sends_csrf_header_and_avoids_inline_code() -> None:
    """AC5: api.js sends the CSRF header on both paths and index.html has no inline code."""
    api = (DEFAULT_STATIC_DIR / "js" / "api.js").read_text(encoding="utf-8")
    assert 'init.headers["X-Viewer-Request"] = "1"' in api
    assert 'setRequestHeader("X-Viewer-Request", "1")' in api
    html = (DEFAULT_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)
    assert "<style" not in html
    assert "style=" not in html
    assert not re.search(r"\son\w+\s*=", html)
