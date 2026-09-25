"""Browser smoke tests: uploads, overview values, npy.js and hash state."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("playwright")

import numpy as np  # noqa: E402
from playwright.sync_api import expect  # noqa: E402

from .conftest import (  # noqa: E402
    DERIVE_TIMEOUT_MS,
    assert_no_console_errors,
    assert_state_roundtrip,
    expected,
    fixtures_dir,
    open_panel,
    save_screenshot,
    upload_bundle,
    upload_expect_error,
    viewer_url,
    wait_panel,
)

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page


def kv_value(root: Locator, label: str) -> str:
    """Return the table.kv value whose row header exactly matches label."""
    rows = root.locator("table.kv tr")
    for index in range(rows.count()):
        row = rows.nth(index)
        if row.locator("th").inner_text() == label:
            return row.locator("td").inner_text()
    raise AssertionError(f"no table.kv row labelled {label!r}")


def _api_bundle_count() -> int:
    """Return the number of bundles known to the viewer API."""
    with urllib.request.urlopen(f"{viewer_url()}/api/bundles") as response:
        body = json.load(response)
    return len(body["bundles"])


def _check_valid_upload(
    page: Page,
    filename: str,
    views_bs: str,
    num_views: str,
    num_base_stations: str,
    schema_version: str,
) -> str:
    """Upload a valid bundle through the UI and check the list row and overview."""
    digest = upload_bundle(page, fixtures_dir() / filename)
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    row = page.locator(f'table.bundles tr[data-digest="{digest}"]')
    expect(row).to_be_visible()
    expect(row.locator("td").nth(3)).to_have_text(views_bs)
    assert row.locator("a").get_attribute("href") == f"#/b/{digest}"
    row.locator("a").click()
    # The router redirects the bare bundle link to the canonical member/panel hash.
    expect(page).to_have_url(
        re.compile(re.escape(f"#/b/{digest}/dataset/overview?v=1") + "$"),
        timeout=DERIVE_TIMEOUT_MS,
    )
    root = wait_panel(page, "overview")
    assert kv_value(root, "Views") == num_views
    assert kv_value(root, "Base stations") == num_base_stations
    assert kv_value(root, "Schema version") == schema_version
    return digest


def test_upload_valid_bundle(page: Page) -> None:
    """Upload the schema-v3 bundle and check the list row and overview values."""
    _check_valid_upload(page, "bundle.zip", "3 / 2", "3", "2", "3")
    assert_no_console_errors(page)
    save_screenshot(page, "test_upload_valid_bundle")


def test_upload_schema_v2_bundle(page: Page) -> None:
    """Upload the schema-v2 bundle and check the list row and overview values."""
    _check_valid_upload(page, "bundle_v2.zip", "3 / 1", "3", "1", "2")
    assert_no_console_errors(page)
    save_screenshot(page, "test_upload_schema_v2_bundle")


def _check_rejected_upload(
    page: Page, filename: str, error_type: str, message_part: str
) -> dict[str, str | None]:
    """Upload a bad archive, check the error view, and check the bundle count is unchanged."""
    count_before = _api_bundle_count()
    result = upload_expect_error(page, fixtures_dir() / filename)
    assert result["type"] == error_type
    message = result["message"]
    assert message is not None and message_part in message
    box = page.locator(".upload-message .status.error")
    expect(box).to_be_visible()
    expect(box.locator("pre.message")).to_contain_text(message_part)
    page.reload()
    expect(page.locator('[data-panel="bundles"]')).to_be_visible()
    table = page.locator("table.bundles")
    empty = page.locator(".bundle-list .status.empty")
    expect(table.or_(empty)).to_be_visible()
    assert page.locator("table.bundles tbody tr[data-digest]").count() == count_before
    assert _api_bundle_count() == count_before
    return result


def test_broken_manifest_is_rejected(page: Page) -> None:
    """Uploading a bundle with an invalid manifest shows a validation error."""
    result = _check_rejected_upload(
        page, "broken_bad_schema_version.zip", "validation", expected()["broken_message"]
    )
    assert result["member"] == "dataset"
    assert_no_console_errors(page, ignore=[r"status of 400"])
    save_screenshot(page, "test_broken_manifest_is_rejected")


def test_unsafe_archive_is_rejected(page: Page) -> None:
    """Uploading an archive with a ../ entry shows an unsafe_archive error."""
    _check_rejected_upload(
        page, "malicious_dotdot.tar.gz", "unsafe_archive", expected()["malicious_member"]
    )
    assert_no_console_errors(page, ignore=[r"status of 400"])
    save_screenshot(page, "test_unsafe_archive_is_rejected")


def _npy_cases() -> dict[str, tuple[np.ndarray, str, tuple[int, int] | None]]:
    """Build the npy.js parser cases keyed by case name."""
    rng = np.random.default_rng(7)
    special = np.array(
        [
            [np.nan, np.inf, -np.inf, -0.0],
            [np.float16(2**-24), np.float16(3e-5), np.float16(65504), np.float16(0.5)],
        ],
        dtype=np.float16,
    )
    return {
        "float16_special": (special, "float16", None),
        "float32_3d": (rng.standard_normal((2, 3, 4)).astype(np.float32), "float32", None),
        "bool_3d": ((np.arange(24).reshape(2, 3, 4) % 2 == 0), "bool", None),
        "uint8_2d": (np.array([[0, 1, 127], [128, 200, 255]], dtype=np.uint8), "uint8", None),
        "int32_1d": (np.array([0, -1, 42, -(2**31), 2**31 - 1], dtype=np.int32), "int32", None),
        "uint32_1d": (np.array([0, 1, 2**31, 2**32 - 1], dtype=np.uint32), "uint32", None),
        "v2_header": (np.arange(6, dtype=np.float32).reshape(2, 3), "float32", (2, 0)),
    }


_NPY_CASES = _npy_cases()

_NPY_PARSE_SNIPPET = """async (b64) => {
  const m = await import('/static/js/npy.js');
  const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
  const parsed = m.parseNpy(bytes);
  const data = Array.from(parsed.data, (v) => (Number.isFinite(v) ? v : String(v)));
  const negZero = Array.from(parsed.data, (v) => Object.is(v, -0));
  return {shape: parsed.shape, dtype: parsed.dtype, data, negZero};
}"""


def _serialise_npy(array: np.ndarray, version: tuple[int, int] | None) -> bytes:
    """Serialise an array to .npy bytes, optionally with a version-2.0 header."""
    buffer = io.BytesIO()
    if version is None:
        np.save(buffer, array)
    else:
        np.lib.format.write_array(buffer, array, version=version)
    return buffer.getvalue()


def _assert_npy_values(case_name: str, array: np.ndarray, result: dict[str, Any]) -> None:
    """Compare parsed values with the source array, decoding non-finite strings."""
    if array.dtype == np.float16:
        wanted = array.ravel().astype(np.float32).astype(np.float64)
    else:
        wanted = array.ravel().astype(np.float64)
    raw = result["data"]
    assert len(raw) == len(wanted), f"{case_name}: {len(raw)} values != {len(wanted)}"
    for index, (got, want) in enumerate(zip(raw, wanted)):
        value = float(got)
        if math.isnan(want):
            assert math.isnan(value), f"{case_name}[{index}]: expected NaN, got {got!r}"
        else:
            assert value == want, f"{case_name}[{index}]: {value!r} != {want!r}"
    assert len(result["negZero"]) == len(wanted)
    if case_name == "float16_special":
        want_flags = [bool(v == 0.0) and bool(np.signbit(v)) for v in wanted]
        assert result["negZero"] == want_flags


@pytest.mark.parametrize("case_name", list(_NPY_CASES))
def test_npy_parser(page: Page, case_name: str) -> None:
    """Feed a NumPy-serialised array through npy.js and compare shape, dtype and values."""
    array, js_dtype, npy_version = _NPY_CASES[case_name]
    payload = base64.b64encode(_serialise_npy(array, npy_version)).decode("ascii")
    page.goto(viewer_url() + "/")
    result = page.evaluate(_NPY_PARSE_SNIPPET, payload)
    assert result["shape"] == list(array.shape)
    assert result["dtype"] == js_dtype
    _assert_npy_values(case_name, array, result)
    assert_no_console_errors(page)
    save_screenshot(page, f"test_npy_parser_{case_name}")


def test_npy_parser_rejects_fortran_order(page: Page) -> None:
    """An F-ordered array makes parseNpy throw a fortran error."""
    array = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert not array.flags["C_CONTIGUOUS"]
    payload = base64.b64encode(_serialise_npy(array, None)).decode("ascii")
    page.goto(viewer_url() + "/")
    message = page.evaluate(
        """async (b64) => {
          const m = await import('/static/js/npy.js');
          const bytes = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0));
          try {
            m.parseNpy(bytes);
          } catch (e) {
            return String((e && e.message) || e);
          }
          return null;
        }""",
        payload,
    )
    assert message is not None, "parseNpy accepted a Fortran-ordered array"
    assert "fortran" in message
    assert_no_console_errors(page)
    save_screenshot(page, "test_npy_parser_rejects_fortran_order")


def test_state_hash_roundtrip(page: Page) -> None:
    """Check parseHash/formatHash round-trips for representative states in the page."""
    page.goto(viewer_url() + "/")
    page.wait_for_function("() => Boolean(window.__viewer)")
    digest = "ab" * 32
    route = {"name": "bundle", "digest": digest, "member": "dataset", "panel": "overview"}
    defaults = {
        "view": None,
        "bs": None,
        "hemisphere": "front",
        "orientation": "rf",
        "path": None,
        "pixel": None,
        "delayBin": None,
        "variant": None,
    }
    partials = [
        {},
        {
            "view": "ue_000001",
            "bs": "bs_001",
            "hemisphere": "back",
            "orientation": "photo",
            "path": 3,
            "pixel": [4, 5],
            "delayBin": 7,
            "variant": "obs",
        },
        {"view": "a b/c?d&e=f#g%h"},
        {"view": "ビュー1"},
        {"path": 0, "pixel": [0, 0], "delayBin": 0},
    ]
    states = [{**defaults, **partial} for partial in partials]
    results = page.evaluate(
        """([route, states]) => states.map((state) => {
          const hash = window.__viewer.formatHash(route, state);
          const parsed = window.__viewer.parseHash(hash);
          return {hash, state: parsed.state, route: parsed.route};
        })""",
        [route, states],
    )
    for full, result in zip(states, results):
        assert result["state"] == full
        assert result["route"] == route
    assert results[0]["hash"] == f"#/b/{digest}/dataset/overview?v=1"
    assert_no_console_errors(page)
    save_screenshot(page, "test_state_hash_roundtrip")


def test_state_restored_after_reload(page: Page, bundle_digests: dict[str, str]) -> None:
    """Set the overview selection, reload, and check the state and highlight persist."""
    open_panel(page, bundle_digests["bundle"], "dataset", "overview")
    assert_state_roundtrip(
        page,
        view="ue_000001",
        bs="bs_001",
        hemisphere="back",
        orientation="photo",
        path=2,
        pixel=[3, 4],
        delayBin=5,
        variant="obs",
    )
    root = wait_panel(page, "overview")
    expect(root.locator("table.pairs")).to_be_visible(timeout=DERIVE_TIMEOUT_MS)
    flags = root.locator("table.pairs tr[data-view][data-bs]").evaluate_all(
        "rows => rows.map((row) => [row.dataset.view, row.dataset.bs,"
        " row.classList.contains('selected')])"
    )
    assert len(flags) > 1
    matches = [row for row in flags if row[0] == "ue_000001" and row[1] == "bs_001"]
    assert len(matches) == 1 and matches[0][2] is True
    assert all(not row[2] for row in flags if not (row[0] == "ue_000001" and row[1] == "bs_001"))
    assert_no_console_errors(page)
    save_screenshot(page, "test_state_restored_after_reload")
