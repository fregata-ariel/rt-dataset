"""FastAPI application factory for the viewer backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.staticfiles import StaticFiles

from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.api.errors import install_error_handlers
from plateau_rt.viewer.api.routes_bundles import router as bundles_router
from plateau_rt.viewer.api.routes_derived import router as derived_router
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def start_eager_derivations(store: Store, digest: str) -> None:
    """Hook called after a new bundle is committed; V0-5b (#40) starts eager derivations here."""
    return None


def _index_response(directory: Path) -> Response:
    """Serve ``directory/index.html`` when present, else the plain-text description."""
    index_path = directory / "index.html"
    if index_path.is_file():
        return FileResponse(index_path, media_type="text/html")
    return PlainTextResponse(
        f"plateau_rt viewer backend {VIEWER_VERSION}: no frontend installed. "
        "The HTTP API is under /api (try /api/health); see docs/viewer.md."
    )


def create_app(settings: ViewerSettings, *, static_dir: Path | None = None) -> FastAPI:
    """Build the viewer FastAPI app, opening the store from ``settings`` once."""
    app = FastAPI(
        title="plateau_rt viewer",
        version=VIEWER_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    store = Store(settings)
    app.state.settings = settings
    app.state.store = store
    app.state.on_bundle_committed = start_eager_derivations
    install_error_handlers(app)
    app.include_router(bundles_router, prefix="/api")
    app.include_router(derived_router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Return the viewer version, index schema version and read-only flag."""
        return {
            "status": "ok",
            "viewer_version": VIEWER_VERSION,
            "read_only": settings.read_only,
            "store_schema_version": store.schema_version,
        }

    directory = DEFAULT_STATIC_DIR if static_dir is None else static_dir

    @app.get("/")
    def index() -> Response:
        """Serve the frontend index when installed, else a plain-text description."""
        return _index_response(directory)

    if directory.is_dir():
        app.mount("/static", StaticFiles(directory=directory), name="static")
    return app


def create_app_from_env() -> FastAPI:
    """Build the viewer app from ``ViewerSettings.from_env()`` (uvicorn ``--factory``)."""
    return create_app(ViewerSettings.from_env())
