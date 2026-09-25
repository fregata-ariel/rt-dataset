"""API error envelope, :class:`ApiError` and the FastAPI exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

ERROR_TYPES = (
    "unsafe_archive",
    "validation",
    "unknown_kind",
    "too_large",
    "insufficient_storage",
    "not_found",
    "bad_params",
    "derive_failed",
    "conflict",
)


class ApiError(Exception):
    """An error that becomes the JSON error envelope."""

    def __init__(self, status: int, type: str, message: str, member: str | None = None) -> None:
        """Validate ``type`` and store the status, type, message and member."""
        if type not in ERROR_TYPES:
            raise ValueError(f"unknown error type {type!r}")
        self.status = status
        self.type = type
        self.message = message
        self.member = member
        super().__init__(message)


def error_body(type: str, message: str, member: str | None = None) -> dict[str, Any]:
    """Return the JSON error envelope body with exactly the three documented keys."""
    return {"error": {"type": type, "member": member, "message": message}}


def error_response(status: int, type: str, message: str, member: str | None = None) -> JSONResponse:
    """Return a JSON response carrying the error envelope."""
    return JSONResponse(status_code=status, content=error_body(type, message, member))


def install_error_handlers(app: FastAPI) -> None:
    """Register the API, HTTP and request-validation exception handlers on ``app``."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status, exc.type, exc.message, exc.member)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return error_response(404, "not_found", str(exc.detail))
        return error_response(exc.status_code, "bad_params", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(400, "bad_params", str(exc))
