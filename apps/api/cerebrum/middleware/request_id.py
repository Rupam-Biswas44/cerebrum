"""
Request ID Middleware

Injects a unique request ID (UUIDv4) into every incoming request if one
isn't already provided. Adds it to the response headers and binds it
to the structlog context for traceability across logs.
"""

import uuid
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Extract existing or generate new request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind to structlog context for all subsequent logs in this request
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Store in request state for use in routes/audit logs
        request.state.request_id = request_id

        # Process the request
        response = await call_next(request)

        # Inject into response headers
        response.headers["X-Request-ID"] = request_id

        # Clear structlog context (optional, but good for cleanliness)
        structlog.contextvars.clear_contextvars()

        return response
