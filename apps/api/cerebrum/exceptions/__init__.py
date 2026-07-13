"""
Exceptions Module

Defines custom application exceptions and configures global exception
handlers for the FastAPI application to ensure consistent JSON error
responses across the entire API.
"""

from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

logger = structlog.get_logger(__name__)


class CerebrumException(Exception):
    """Base exception for all custom application errors."""
    def __init__(self, message: str, status_code: int = 500, code: str = "internal_error", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class AuthenticationError(CerebrumException):
    """Raised when authentication fails or is missing."""
    def __init__(self, message: str = "Authentication failed", details: dict[str, Any] | None = None):
        super().__init__(message, status_code=401, code="unauthorized", details=details)


class AuthorizationError(CerebrumException):
    """Raised when an authenticated user lacks required permissions."""
    def __init__(self, message: str = "Permission denied", details: dict[str, Any] | None = None):
        super().__init__(message, status_code=403, code="forbidden", details=details)


class NotFoundError(CerebrumException):
    """Raised when a requested resource is not found."""
    def __init__(self, resource: str, resource_id: str | None = None):
        msg = f"{resource} not found"
        if resource_id:
            msg += f" (id: {resource_id})"
        super().__init__(msg, status_code=404, code="not_found", details={"resource": resource, "id": resource_id})


class ValidationError(CerebrumException):
    """Raised when business logic validation fails."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=422, code="validation_error", details=details)


def configure_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""

    @app.exception_handler(CerebrumException)
    async def cerebrum_exception_handler(request: Request, exc: CerebrumException) -> ORJSONResponse:
        logger.warning(
            "api.error.cerebrum",
            error_code=exc.code,
            error_msg=exc.message,
            path=request.url.path,
        )
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> ORJSONResponse:
        logger.info("api.error.validation", errors=exc.errors(), path=request.url.path)
        return ORJSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "request_validation_error",
                    "message": "Invalid request parameters",
                    "details": exc.errors(),
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> ORJSONResponse:
        logger.warning("api.error.http", status_code=exc.status_code, detail=exc.detail, path=request.url.path)
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "details": {},
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
        logger.error("api.error.unhandled", error=str(exc), path=request.url.path, exc_info=exc)
        return ORJSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "details": {},
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )
