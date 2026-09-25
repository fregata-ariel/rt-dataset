"""FastAPI application factory for the viewer backend."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.api.errors import install_error_handlers
from plateau_rt.viewer.api.routes_bundles import router as bundles_router
from plateau_rt.viewer.api.routes_derived import router as derived_router
from plateau_rt.viewer.api.routes_jobs import router as jobs_router
from plateau_rt.viewer.api.security import install_security
from plateau_rt.viewer.api.static_assets import install_static_routes
from plateau_rt.viewer.jobs import JobManager
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store

DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def eager_hook(jobs: JobManager) -> Callable[[Store, str], None]:
    """Return the bundle-committed hook that queues a bundle's missing eager derivations."""

    def _hook(store: Store, digest: str) -> None:
        jobs.submit_eager(digest)

    return _hook


def create_app(settings: ViewerSettings, *, static_dir: Path | None = None) -> FastAPI:
    """Build the viewer FastAPI app, opening the store from ``settings`` once."""
    store = Store(settings)
    jobs = JobManager(store)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Shut the job manager down after the server stops."""
        yield
        jobs.shutdown()

    app = FastAPI(
        title="plateau_rt viewer",
        version=VIEWER_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.jobs = jobs
    app.state.on_bundle_committed = eager_hook(jobs)
    install_error_handlers(app)
    install_security(app, settings)
    app.include_router(bundles_router, prefix="/api")
    app.include_router(derived_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")

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

    install_static_routes(app, directory)
    jobs.recover()
    return app


def create_app_from_env() -> FastAPI:
    """Build the viewer app from ``ViewerSettings.from_env()`` (uvicorn ``--factory``)."""
    return create_app(ViewerSettings.from_env())
