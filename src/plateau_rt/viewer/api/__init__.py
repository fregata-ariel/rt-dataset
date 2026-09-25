"""HTTP API for the plateau_rt viewer (FastAPI application factory)."""

from __future__ import annotations

from plateau_rt.viewer.api.app import create_app, create_app_from_env

__all__ = ["create_app", "create_app_from_env"]
