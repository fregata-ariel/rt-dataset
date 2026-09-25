"""Browser tests for the viewer security baseline (headers, CSP and XSS rendering)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("playwright")

from playwright.sync_api import expect  # noqa: E402

from .conftest import (  # noqa: E402
    CSP_CONSOLE_PATTERN,
    DERIVE_TIMEOUT_MS,
    assert_no_console_errors,
    csp_violations,
    expected,
    fixtures_dir,
    open_panel,
    save_screenshot,
    upload_expect_error,
    viewer_url,
)
from .test_panels import PANEL_CASES, _panel_params  # noqa: E402

if TYPE_CHECKING:
    from playwright.sync_api import Page

_DIGEST_RE = re.compile(r"#/b/([0-9a-f]{64})")

_VENDOR_PROBE = """async (name) => {
  const vendor = await import("/static/js/vendor.js");
  const dom = await import("/static/js/dom.js");
  const escaped = dom.escapeHtml(name);
  const el = document.createElement("div");
  el.style.width = "400px";
  el.style.height = "300px";
  document.body.appendChild(el);
  const Plotly = await vendor.loadPlotly();
  await Plotly.newPlot(el, [
    {x: [0, 1], y: [0, 1], type: "scatter", mode: "lines+markers",
     text: [escaped], hovertemplate: "%{text}<extra></extra>"},
    {z: [[0, 1], [1, 0]], type: "heatmap"},
  ], {title: {text: escaped}}, {displayModeBar: false});
  Plotly.Fx.hover(el, [{curveNumber: 0, pointNumber: 0}]);
  await Plotly.toImage(el, {format: "png", width: 100, height: 100});
  const { THREE, OrbitControls } = await vendor.loadThree();
  const canvas = document.createElement("canvas");
  const renderer = new THREE.WebGLRenderer({canvas});
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 0, 3);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.update();
  const pixels = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]);
  const texture = new THREE.DataTexture(pixels, 2, 2, THREE.RGBAFormat);
  texture.needsUpdate = true;
  const scene = new THREE.Scene();
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({map: texture}),
  );
  scene.add(mesh);
  renderer.setSize(64, 64);
  renderer.render(scene, camera);
  return {ok: true, three: THREE.REVISION};
}"""


def test_csp_header_on_index(page: Page) -> None:
    """The index response carries the baseline CSP with script-src and frame-ancestors."""
    response = page.goto(viewer_url() + "/")
    assert response is not None
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_csp_collector_sees_violations(page: Page) -> None:
    """The collector records real violations, so the zero-count checks are not vacuous."""
    page.goto(viewer_url() + "/#/")
    page.evaluate(
        """() => {
          const script = document.createElement("script");
          script.textContent = "window.__xss = 3";
          document.body.appendChild(script);
          const div = document.createElement("div");
          div.setAttribute("onclick", "window.__xss = 4");
          document.body.appendChild(div);
          div.click();
        }"""
    )
    assert csp_violations(page)
    assert page.evaluate("() => typeof window.__xss") == "undefined"


def test_xss_bundle_name_is_text(page: Page) -> None:
    """A bundle name with markup renders as text in the list and the header, without CSP hits."""
    page.goto(viewer_url() + "/#/")
    page.locator("input[type=file]").set_input_files(fixtures_dir() / expected()["xss_bundle"])
    name = expected()["xss_bundle_name"]
    page.locator(".upload-controls input[type=text]").fill(name)
    page.get_by_role("button", name="Upload", exact=True).click()
    ok = page.locator(".upload-message .status.ok")
    expect(ok).to_be_visible(timeout=DERIVE_TIMEOUT_MS)
    href = ok.locator("a").get_attribute("href") or ""
    match = _DIGEST_RE.search(href)
    assert match is not None, f"upload link has no digest: {href!r}"
    digest = match.group(1)
    expect(page.locator(f'table.bundles tr[data-digest="{digest}"]')).to_be_visible(
        timeout=DERIVE_TIMEOUT_MS
    )
    row_link = page.locator(f'table.bundles tr[data-digest="{digest}"] a')
    assert row_link.inner_text() == name
    assert name in ok.inner_text()
    assert page.locator("main img").count() == 0
    assert page.evaluate("() => typeof window.__xss") == "undefined"
    assert csp_violations(page) == []
    assert_no_console_errors(page, ignore=[CSP_CONSOLE_PATTERN])

    open_panel(page, digest, "dataset", "overview", view="ue_000001", bs="bs_000")
    assert page.locator("main .bundle-name").inner_text() == name
    assert page.locator("main img").count() == 0
    assert page.evaluate("() => typeof window.__xss") == "undefined"
    assert csp_violations(page) == []
    assert_no_console_errors(page, ignore=[CSP_CONSOLE_PATTERN])
    save_screenshot(page, "security-xss-name")


def test_xss_manifest_error_is_text(page: Page) -> None:
    """An XSS payload in a manifest error string renders as text without CSP hits."""
    result = upload_expect_error(page, fixtures_dir() / "broken_script_in_message.zip")
    assert result["type"] == "validation"
    assert result["member"] == "dataset"
    message = result["message"] or ""
    assert expected()["xss_broken_message"] in message
    assert page.locator(".upload-message script").count() == 0
    assert page.locator(".upload-message img").count() == 0
    assert page.evaluate("() => typeof window.__xss") == "undefined"
    assert csp_violations(page) == []
    assert_no_console_errors(page, ignore=[r"status of 400", CSP_CONSOLE_PATTERN])
    save_screenshot(page, "security-xss-error")


@pytest.mark.parametrize(("panel_id", "state", "bundle"), _panel_params())
def test_all_panels_have_no_csp_violations(
    page: Page,
    bundle_digests: dict[str, str],
    panel_id: str,
    state: dict[str, Any],
    bundle: str,
) -> None:
    """Every panel opens under the baseline CSP without a violation or a console error."""
    assert panel_id in {case[0] for case in PANEL_CASES}
    if panel_id == "bundles":
        open_panel(page, None, None, "bundles")
    else:
        open_panel(page, bundle_digests[bundle], "dataset", panel_id, **state)
    assert csp_violations(page) == []
    assert_no_console_errors(page)


def test_vendor_libraries_run_under_csp(page: Page) -> None:
    """Plotly strict and three.js run under the CSP with no extra allowances."""
    page.goto(viewer_url() + "/")
    page.wait_for_function("() => Boolean(window.__viewer)")
    result = page.evaluate(_VENDOR_PROBE, expected()["xss_bundle_name"])
    assert result["ok"] is True
    assert result["three"]
    assert page.evaluate("() => typeof window.__xss") == "undefined"
    assert csp_violations(page) == []
    assert_no_console_errors(page)
