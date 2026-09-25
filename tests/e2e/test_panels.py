"""One open-check per panel.

To cover a new panel, append one ``(panel_id, state)`` line to PANEL_CASES (and, when
the panel root needs a different probe, one entry to PANEL_ROOT_CHECK), then run
``scripts/ci/build-images.sh viewer viewer-e2e && scripts/ci/run-viewer-e2e.sh -k <panel_id>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("playwright")

from playwright.sync_api import expect  # noqa: E402

from .conftest import (  # noqa: E402
    assert_no_console_errors,
    assert_state_roundtrip,
    open_panel,
    save_screenshot,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

PANEL_CASES: list[tuple[str, dict[str, Any]]] = [
    ("bundles", {}),
    ("overview", {"view": "ue_000001", "bs": "bs_000"}),
]
"""One ``(panel_id, state)`` entry per panel; later panel issues append one line."""

PANEL_ROOT_CHECK: dict[str, str] = {
    "bundles": "table.bundles",
    "overview": "h2",
}
"""Panel id to a CSS selector that must be visible inside the panel root."""


def _panel_params() -> list[Any]:
    """Expand PANEL_CASES over both bundles, running the home panel only once."""
    params = []
    for panel_id, state in PANEL_CASES:
        if panel_id == "bundles":
            params.append(pytest.param(panel_id, state, "bundle", id=panel_id))
        else:
            for bundle in ("bundle", "bundle_v2"):
                params.append(pytest.param(panel_id, state, bundle, id=f"{panel_id}-{bundle}"))
    return params


@pytest.mark.parametrize(("panel_id", "state", "bundle"), _panel_params())
def test_panel_opens(
    page: Page, bundle_digests: dict[str, str], panel_id: str, state: dict[str, Any], bundle: str
) -> None:
    """Open one panel, check its root content, round-trip its state and screenshot it."""
    if panel_id == "bundles":
        root = open_panel(page, None, None, "bundles")
    else:
        root = open_panel(page, bundle_digests[bundle], "dataset", panel_id, **state)
    expect(root.locator(PANEL_ROOT_CHECK[panel_id]).first).to_be_visible()
    if panel_id == "bundles":
        rows = root.locator("table.bundles tbody tr[data-digest]")
        expect(rows.first).to_be_visible()
        expect(rows.nth(1)).to_be_visible()
    assert_state_roundtrip(page, **state)
    assert_no_console_errors(page)
    save_screenshot(page, f"panel-{panel_id}-{bundle}")
