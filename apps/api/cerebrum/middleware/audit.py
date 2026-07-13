"""
Audit Log Middleware

Captures every mutating request (POST, PUT, PATCH, DELETE) and asynchronously
writes an audit log to the database. Essential for enterprise compliance.
"""

import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import async_session_factory
from models.domain import AuditLog
import structlog

logger = structlog.get_logger(__name__)


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # We only want to audit state-mutating requests
        # Or requests that are explicitly marked for auditing
        is_mutating = request.method in ("POST", "PUT", "PATCH", "DELETE")
        
        response = await call_next(request)

        # Skip auditing if not a mutating request, or if it's a healthcheck/metric
        if not is_mutating or request.url.path in ("/health", "/metrics", "/api/v1/auth/token"):
            return response

        # Write audit log asynchronously in the background
        # We don't want to block the response to the user
        from fastapi import BackgroundTasks
        if not hasattr(request.state, "background_tasks"):
            request.state.background_tasks = BackgroundTasks()

        user_id = getattr(request.state, "user_id", None)
        request_id = getattr(request.state, "request_id", None)
        
        # Fire and forget the audit log write
        await self._log_audit(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            user_id=user_id,
            request_id=request_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        return response

    async def _log_audit(
        self,
        method: str,
        path: str,
        status_code: int,
        user_id: Any | None,
        request_id: str | None,
        ip_address: str | None,
        user_agent: str | None,
    ) -> None:
        """Writes the actual audit log entry to the DB."""
        if async_session_factory is None:
            # DB not initialized
            return

        try:
            async with async_session_factory() as session:
                log_entry = AuditLog(
                    user_id=user_id,
                    action=f"{method} {path}",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    request_id=request_id,
                    status_code=status_code,
                    details={"path": path, "method": method},
                )
                session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.error("audit_log.failed", error=str(e), path=path)
