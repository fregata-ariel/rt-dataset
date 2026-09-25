"""Tests for the viewer static frontend shell (build-hashed immutable assets)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

from viewer_client import viewer_client

from plateau_rt.viewer.api import create_app
from plateau_rt.viewer.api.app import DEFAULT_STATIC_DIR
from plateau_rt.viewer.settings import ViewerSettings

with warnings.catch_warnings():
    # Starlette 1.x warns that its TestClient will move from httpx to httpx2.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

MiB = 1 << 20
BUILD_RE = re.compile(r"^[0-9a-f]{12}$")
FROM_RE = re.compile(r"""(?:import|export)[^'"]*?from\s*["']([^"']+)["']""")
DYNAMIC_IMPORT_RE = re.compile(r"""import\(\s*["']([^"']+)["']\s*\)""")
SIDE_EFFECT_IMPORT_RE = re.compile(r"""import\s*["']([^"']+)["']""")


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


def make_app(tmp_path: Path, static_dir: Path, **overrides: Any) -> Any:
    """Build an app serving ``static_dir`` with the eager hook disabled."""
    app = create_app(_settings(tmp_path, **overrides), static_dir=static_dir)
    app.state.on_bundle_committed = lambda store, digest: None
    return app


def make_client(tmp_path: Path, static_dir: Path) -> tuple[Any, TestClient]:
    """Build an app and a TestClient serving ``static_dir``."""
    app = make_app(tmp_path, static_dir)
    return app, viewer_client(app)


def static_files() -> list[str]:
    """Return every file relpath under the real static dir, sorted."""
    return sorted(
        path.relative_to(DEFAULT_STATIC_DIR).as_posix()
        for path in DEFAULT_STATIC_DIR.rglob("*")
        if path.is_file()
    )


def expected_media_type(relpath: str) -> str | None:
    """Return the media-type prefix asserted for ``relpath``, or None."""
    if relpath.endswith((".js", ".mjs")):
        return "text/javascript"
    if relpath.endswith(".css"):
        return "text/css"
    if relpath.endswith(".json"):
        return "application/json"
    return None


def test_index_served_no_cache(tmp_path: Path) -> None:
    """GET / serves index.html with no-cache and the versioned asset URLs."""
    app, client = make_client(tmp_path, DEFAULT_STATIC_DIR)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "no-cache" in response.headers["cache-control"]
    build = app.state.static_build
    assert BUILD_RE.fullmatch(build) is not None
    assert "__BUILD__" not in response.text
    assert f"/static/{build}/js/app.js" in response.text
    assert f"/static/{build}/css/viewer.css" in response.text


def test_every_static_file_served_immutable(tmp_path: Path) -> None:
    """Every static file is served versioned with immutable caching and an ETag."""
    app, client = make_client(tmp_path, DEFAULT_STATIC_DIR)
    build = app.state.static_build
    relpaths = [path for path in static_files() if path != "index.html"]
    assert len(relpaths) > 0
    for relpath in relpaths:
        url = f"/static/{build}/{relpath}"
        response = client.get(url)
        assert response.status_code == 200, relpath
        assert response.content == (DEFAULT_STATIC_DIR / relpath).read_bytes(), relpath
        assert "immutable" in response.headers["cache-control"], relpath
        assert response.headers["x-content-type-options"] == "nosniff", relpath
        expected = expected_media_type(relpath)
        if expected is not None:
            assert response.headers["content-type"].startswith(expected), relpath
        sha256 = hashlib.sha256(response.content).hexdigest()
        assert response.headers["etag"] == f'"{sha256}"', relpath
        again = client.get(url, headers={"If-None-Match": f'"{sha256}"'})
        assert again.status_code == 304, relpath
        assert again.content == b"", relpath


def test_unversioned_static_no_cache(tmp_path: Path) -> None:
    """GET /static/js/npy.js serves the file unversioned with no-cache."""
    _app, client = make_client(tmp_path, DEFAULT_STATIC_DIR)
    response = client.get("/static/js/npy.js")
    assert response.status_code == 200
    assert "no-cache" in response.headers["cache-control"]
    assert response.content == (DEFAULT_STATIC_DIR / "js" / "npy.js").read_bytes()


def test_static_rejects_traversal_and_unknown(tmp_path: Path) -> None:
    """Traversal, wrong builds, index under /static and new files all answer 404."""
    app, client = make_client(tmp_path, DEFAULT_STATIC_DIR)
    build = app.state.static_build
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    paths = [
        f"/static/{build}/index.html",
        "/static/index.html",
        "/static/000000000000/js/app.js",
        f"/static/{build}/../index.html",
        f"/static/{build}/%2e%2e/%2e%2e/api/app.py",
        f"/static/{build}/js/../../../../pyproject.toml",
        "/static/%2e%2e/outside.txt",
        f"/static/{build}/js/%00app.js",
        f"/static/{build}/js/app.js%2f..",
        "/static/nope-missing.js",
    ]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.json()["error"]["type"] == "not_found", path


def test_build_changes_when_a_file_changes(tmp_path: Path) -> None:
    """Editing, adding or renaming a static file changes the build (snapshot reads)."""
    static = tmp_path / "static"
    shutil.copytree(DEFAULT_STATIC_DIR, static)
    app_a = make_app(tmp_path / "a", static)
    app_a2 = make_app(tmp_path / "a2", static)
    build_a = app_a.state.static_build
    assert BUILD_RE.fullmatch(build_a) is not None
    assert app_a2.state.static_build == build_a
    client_a = viewer_client(app_a)
    old_bytes = client_a.get(f"/static/{build_a}/js/strings.js").content
    assert old_bytes == (static / "js" / "strings.js").read_bytes()

    with (static / "js" / "strings.js").open("ab") as handle:
        handle.write(b"\n// changed\n")
    app_b = make_app(tmp_path / "b", static)
    build_b = app_b.state.static_build
    assert build_b != build_a
    client_b = viewer_client(app_b)
    index_b = client_b.get("/")
    assert f"/static/{build_b}/js/app.js" in index_b.text
    new_bytes = client_b.get(f"/static/{build_b}/js/strings.js").content
    assert new_bytes == (static / "js" / "strings.js").read_bytes()
    assert new_bytes != old_bytes
    assert client_b.get(f"/static/{build_a}/js/app.js").status_code == 404
    assert client_a.get(f"/static/{build_a}/js/strings.js").content == old_bytes

    late = static / "js" / "late.js"
    late.write_bytes(b"export const late = 1;\n")
    assert client_b.get(f"/static/{build_b}/js/late.js").status_code == 404
    assert client_b.get("/static/js/late.js").status_code == 404
    late.unlink()

    vendor_extra = static / "vendor" / "extra.txt"
    vendor_extra.write_bytes(b"extra\n")
    app_c = make_app(tmp_path / "c", static)
    assert app_c.state.static_build not in (build_a, build_b)


def test_vendor_sha256_matches_vendor_json(tmp_path: Path) -> None:
    """VENDOR.json lists every vendor file exactly once with matching sha256."""
    del tmp_path
    record = json.loads((DEFAULT_STATIC_DIR / "vendor" / "VENDOR.json").read_text())
    entries = record["files"]
    on_disk = sorted(
        path.relative_to(DEFAULT_STATIC_DIR / "vendor").as_posix()
        for path in (DEFAULT_STATIC_DIR / "vendor").rglob("*")
        if path.is_file() and path.name != "VENDOR.json"
    )
    listed = [f"vendor/{entry['path']}" for entry in entries]
    assert sorted(f"vendor/{path}" for path in on_disk) == sorted(
        path for path in listed if path != "vendor/VENDOR.json"
    )
    assert len(set(listed)) == len(listed)
    listed_set = set(listed)
    for entry in entries:
        assert entry["package"]
        assert entry["version"]
        assert entry["source_url"].startswith("https://")
        assert entry["license"]
        assert f"vendor/{entry['license_file']}" in listed_set
        target = DEFAULT_STATIC_DIR / "vendor" / entry["path"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]
    for required in (
        "three/three.module.js",
        "three/three.core.js",
        "three/OrbitControls.js",
        "plotly/plotly-strict.min.js",
    ):
        assert f"vendor/{required}" in listed_set
    controls = (DEFAULT_STATIC_DIR / "vendor" / "three" / "OrbitControls.js").read_text()
    assert "from 'three'" not in controls
    assert "./three.module.js" in controls


def js_specifiers(path: Path) -> list[str]:
    """Return every static import/export specifier of one JS file."""
    text = path.read_text(encoding="utf-8")
    found = FROM_RE.findall(text)
    found.extend(DYNAMIC_IMPORT_RE.findall(text))
    found.extend(SIDE_EFFECT_IMPORT_RE.findall(text))
    return found


def test_js_relative_imports_resolve(tmp_path: Path) -> None:
    """Every JS import specifier is relative and resolves inside the static dir."""
    del tmp_path
    js_files = sorted((DEFAULT_STATIC_DIR / "js").rglob("*.js"))
    assert len(js_files) > 0
    for path in js_files:
        for spec in js_specifiers(path):
            assert spec.startswith(("./", "../")), f"{path}: {spec}"
            assert "/static" not in spec, f"{path}: {spec}"
            target = (path.parent / spec).resolve()
            assert str(target).startswith(str(DEFAULT_STATIC_DIR.resolve())), f"{path}: {spec}"
            assert target.is_file(), f"{path}: {spec}"


def test_js_forbidden_apis(tmp_path: Path) -> None:
    """Our JS avoids markup sinks, dialogs and inline code in index.html."""
    del tmp_path
    forbidden = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "confirm(",
        "prompt(",
        "alert(",
    )
    for path in sorted((DEFAULT_STATIC_DIR / "js").rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: {token}"
    html = (DEFAULT_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)
    assert "<style" not in html
    assert "style=" not in html
    assert not re.search(r"\son\w+\s*=", html)


def test_panels_registered(tmp_path: Path) -> None:
    """Every panel module is registered with an id, kinds and a default export."""
    del tmp_path
    panels_dir = DEFAULT_STATIC_DIR / "js" / "panels"
    registry = (panels_dir / "registry.js").read_text(encoding="utf-8")
    modules = sorted(path.name for path in panels_dir.glob("*.js") if path.name != "registry.js")
    assert len(modules) > 0
    for name in modules:
        assert f'from "./{name}"' in registry, name
        text = (panels_dir / name).read_text(encoding="utf-8")
        assert "export default" in text, name
    overview = (panels_dir / "overview.js").read_text(encoding="utf-8")
    assert 'kinds: ["rf_dataset"]' in overview


def test_strings_module_is_the_only_ui_text_source(tmp_path: Path) -> None:
    """UI strings live in strings.js, which panels and the router import."""
    del tmp_path
    strings = (DEFAULT_STATIC_DIR / "js" / "strings.js").read_text(encoding="utf-8")
    assert "export const S" in strings
    for path in (
        DEFAULT_STATIC_DIR / "js" / "panels" / "bundles.js",
        DEFAULT_STATIC_DIR / "js" / "panels" / "overview.js",
    ):
        assert 'from "../strings.js"' in path.read_text(encoding="utf-8"), path
    assert 'from "./strings.js"' in (DEFAULT_STATIC_DIR / "js" / "app.js").read_text()


def test_no_physics_in_js(tmp_path: Path) -> None:
    """Guard: our JS never computes physics or conventions (review does the real check)."""
    del tmp_path
    forbidden = (
        "Math.atan2",
        "Math.asin",
        "Math.acos",
        "Math.sin",
        "Math.cos",
        "Math.log10",
        "Math.log",
        "Math.hypot",
        "% unambiguous",
        "Math.PI",
        "mirror",
        "flip",
    )
    for path in sorted((DEFAULT_STATIC_DIR / "js").rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: {token}"
