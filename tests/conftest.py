"""Shared pytest configuration for the viewer golden tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--update-viewer-golden`` option."""
    parser.addoption(
        "--update-viewer-golden",
        action="store_true",
        default=False,
        help="rewrite tests/viewer_golden/derivers.json from the current derivers",
    )
