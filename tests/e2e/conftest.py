"""Shared fixtures and helpers for the viewer browser smoke tests."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from playwright.sync_api import Locator, Page

#: Timeout for anything that waits on a derivation.
DERIVE_TIMEOUT_MS = 30_000

#: Directory for screenshots and other test outputs.
REPORT_DIR = Path(os.environ.get("VIEWER_E2E_REPORT_DIR", "ci-reports/viewer-e2e"))

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SCREENSHOT_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")

_console_errors: dict[int, list[str]] = {}
"""Console/page errors recorded per page id by the ``page`` fixture."""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip e2e tests when playwright is missing or VIEWER_URL is unset."""
    missing = importlib.util.find_spec("playwright") is None
    no_url = not os.environ.get("VIEWER_URL")
    if not missing and not no_url:
        return
    reason = "playwright is not installed" if missing else "VIEWER_URL is not set"
    root = Path(__file__).resolve().parent
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        node_path = Path(str(item.path)).resolve()
        if node_path == root or root in node_path.parents:
            item.add_marker(marker)


def viewer_url() -> str:
    """Return VIEWER_URL without a trailing slash."""
    return os.environ["VIEWER_URL"].rstrip("/")


def fixtures_dir() -> Path:
    """Return the directory holding the e2e fixture archives."""
    return Path(os.environ.get("VIEWER_E2E_FIXTURES", "ci-reports/viewer-e2e/fixtures"))


def expected() -> dict[str, str]:
    """Load the expected error texts written by the fixture script."""
    with open(fixtures_dir() / "expected.json", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def page(page: Page) -> Iterator[Page]:
    """Capture console errors and page errors for assert_no_console_errors."""
    errors: list[str] = []
    _console_errors[id(page)] = errors

    def _on_console(message: Any) -> None:
        """Record console messages of type error."""
        if message.type == "error":
            errors.append(f"console: {message.text}")

    def _on_page_error(exc: Any) -> None:
        """Record uncaught page exceptions."""
        errors.append(f"pageerror: {exc}")

    page.on("console", _on_console)
    page.on("pageerror", _on_page_error)
    try:
        yield page
    finally:
        _console_errors.pop(id(page), None)


def _jsonable(state: dict[str, Any]) -> dict[str, Any]:
    """Convert tuples to lists so the state survives the evaluate round-trip."""
    return {
        key: (list(value) if isinstance(value, tuple) else value) for key, value in state.items()
    }


def api_upload(path: Path, name: str | None = None) -> str:
    """Upload an archive through the HTTP API and return its digest."""
    upload_name = name if name is not None else path.stem
    url = f"{viewer_url()}/api/bundles/upload?name={urllib.parse.quote(upload_name)}"
    request = urllib.request.Request(
        url,
        data=path.read_bytes(),
        method="PUT",
        headers={"Content-Type": "application/octet-stream", "X-Viewer-Request": "1"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise AssertionError(
            f"API upload of {path} failed: {exc.read().decode('utf-8', 'replace')}"
        ) from exc
    digest = str(body["digest"])
    if _DIGEST_RE.fullmatch(digest) is None:
        raise AssertionError(f"API upload of {path} returned a bad digest: {digest!r}")
    return digest


def upload_bundle(page: Page, path: Path) -> str:
    """Upload an archive through the home panel UI and return its digest."""
    from playwright.sync_api import expect

    page.goto(viewer_url() + "/#/")
    expect(page.locator('[data-panel="bundles"]')).to_be_visible()
    page.locator("input[type=file]").set_input_files(path)
    page.get_by_role("button", name="Upload", exact=True).click()
    ok = page.locator(".upload-message .status.ok")
    err = page.locator(".upload-message .status.error")
    expect(ok.or_(err)).to_be_visible(timeout=DERIVE_TIMEOUT_MS)
    if err.is_visible():
        err_type = err.locator(".error-type").inner_text()
        message = err.locator("pre.message").inner_text()
        raise AssertionError(f"upload of {path} failed: {err_type}: {message}")
    href = ok.locator("a").get_attribute("href") or ""
    match = re.search(r"#/b/([0-9a-f]{64})", href)
    if match is None:
        raise AssertionError(f"upload of {path} produced a bad link href: {href!r}")
    digest = match.group(1)
    expect(page.locator(f'table.bundles tr[data-digest="{digest}"]')).to_be_visible(
        timeout=DERIVE_TIMEOUT_MS
    )
    return digest


def upload_expect_error(page: Page, path: Path) -> dict[str, str | None]:
    """Upload an archive through the UI, expecting rejection, and return its error fields."""
    from playwright.sync_api import expect

    page.goto(viewer_url() + "/#/")
    expect(page.locator('[data-panel="bundles"]')).to_be_visible()
    page.locator("input[type=file]").set_input_files(path)
    page.get_by_role("button", name="Upload", exact=True).click()
    box = page.locator(".upload-message .status.error")
    ok = page.locator(".upload-message .status.ok")
    expect(box.or_(ok)).to_be_visible(timeout=DERIVE_TIMEOUT_MS)
    if ok.is_visible():
        raise AssertionError(f"upload of {path} unexpectedly succeeded")
    member_loc = box.locator(".error-member")
    return {
        "type": box.locator(".error-type").inner_text(),
        "member": member_loc.inner_text() if member_loc.count() > 0 else None,
        "message": box.locator("pre.message").inner_text(),
    }


def open_panel(
    page: Page, digest: str | None, member: str | None, panel_id: str, **state: Any
) -> Locator:
    """Open a panel by hash, wait until its derivation finished, and return its root."""

    if digest is None:
        if member is not None or panel_id != "bundles" or state:
            raise ValueError("the home route takes no member, panel or state")
        page.goto(viewer_url() + "/#/")
    else:
        page.goto(viewer_url() + "/")
        page.wait_for_function("() => Boolean(window.__viewer)")
        route = {"name": "bundle", "digest": digest, "member": member, "panel": panel_id}
        target_hash = page.evaluate(
            "([route, patch]) => window.__viewer.formatHash("
            "route, {...window.__viewer.store.get(), ...patch})",
            [route, _jsonable(state)],
        )
        page.goto(viewer_url() + "/" + str(target_hash))
    return wait_panel(page, panel_id)


def wait_panel(page: Page, panel_id: str) -> Locator:
    """Wait until the mounted panel finished loading without an error and return its root."""
    from playwright.sync_api import expect

    root = page.locator(f'main [data-panel="{panel_id}"]')
    expect(root).to_be_visible(timeout=DERIVE_TIMEOUT_MS)
    expect(root.locator(".status.loading")).to_have_count(0, timeout=DERIVE_TIMEOUT_MS)
    error_box = root.locator(".status.error")
    try:
        expect(error_box).to_have_count(0)
    except AssertionError as exc:
        raise AssertionError(
            f"panel {panel_id} shows an error: {error_box.first.inner_text()}"
        ) from exc
    return root


def assert_no_console_errors(page: Page, ignore: Sequence[str] = ()) -> None:
    """Assert no console/page errors were recorded, except ignored console patterns."""
    bad = []
    for entry in _console_errors.get(id(page), []):
        if entry.startswith("pageerror:"):
            bad.append(entry)
        elif not any(re.search(pattern, entry) for pattern in ignore):
            bad.append(entry)
    if bad:
        raise AssertionError("unexpected console errors:\n" + "\n".join(bad))


def save_screenshot(page: Page, name: str) -> Path:
    """Save a full-page screenshot to the report dir and return its path."""
    if _SCREENSHOT_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"bad screenshot name: {name!r}")
    out = REPORT_DIR / "screenshots" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=out, full_page=True)
    return out


def assert_state_roundtrip(page: Page, **state: Any) -> None:
    """Set state via the store, check the hash round-trips, reload and check it persists.

    On the home route the state is not written to the URL, so only an empty
    state round-trips there; that is expected behaviour.
    """
    from playwright.sync_api import expect

    patch = _jsonable(state)
    page.evaluate("patch => window.__viewer.store.set(patch)", patch)
    before = page.evaluate("() => window.__viewer.store.get()")
    for key, value in patch.items():
        current = before[key]
        want = list(value) if isinstance(value, list) else value
        current = list(current) if isinstance(current, list) else current
        assert current == want, f"state {key}: {current!r} != {want!r}"
    url_hash = str(page.evaluate("() => window.location.hash"))
    parsed = page.evaluate("hash => window.__viewer.parseHash(hash)", url_hash)
    assert parsed["state"] == before, f"parsed state {parsed['state']!r} != {before!r}"
    canonical = page.evaluate(
        "hash => { const p = window.__viewer.parseHash(hash);"
        " return window.__viewer.formatHash(p.route, p.state); }",
        url_hash,
    )
    assert canonical == url_hash, f"reformatted hash {canonical!r} != {url_hash!r}"
    page.reload()
    page.wait_for_function("() => Boolean(window.__viewer)")
    expect(page.locator("main [data-panel]").first).to_be_visible()
    assert str(page.evaluate("() => window.location.hash")) == url_hash
    assert page.evaluate("() => window.__viewer.store.get()") == before


@pytest.fixture(scope="session")
def bundle_digests() -> dict[str, str]:
    """Upload the valid fixtures once per session via the HTTP API."""
    return {
        "bundle": api_upload(fixtures_dir() / "bundle.zip"),
        "bundle_v2": api_upload(fixtures_dir() / "bundle_v2.zip"),
    }
