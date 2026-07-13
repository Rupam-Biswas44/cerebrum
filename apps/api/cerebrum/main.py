"""
Cerebrum API — Main Application Entry Point

This module creates the FastAPI application with all middleware,
routers, event handlers, and observability configured.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from cerebrum.config import get_settings
from cerebrum.core.database import create_db_pool, dispose_db_pool
from cerebrum.core.cache import create_redis_pool, close_redis_pool
from cerebrum.core.observability import configure_telemetry, configure_logging
from cerebrum.routers import (
    auth,
    users,
    projects,
    datasets,
    agents,
    tasks,
    ml,
    visualizations,
    reports,
    health,
    metrics,
)
from cerebrum.middleware.rate_limit import RateLimitMiddleware
from cerebrum.middleware.request_id import RequestIDMiddleware
from cerebrum.middleware.audit import AuditLogMiddleware

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Handles startup (resource initialization) and shutdown (cleanup).
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info(
        "cerebrum.startup",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )

    # Initialize database connection pool
    await create_db_pool()
    logger.info("database.pool.ready")

    # Initialize Redis connection pool
    await create_redis_pool()
    logger.info("redis.pool.ready")

    # Configure OpenTelemetry
    configure_telemetry(service_name=settings.OTEL_SERVICE_NAME)
    logger.info("telemetry.ready")

    logger.info("cerebrum.ready", host=settings.APP_HOST, port=settings.APP_PORT)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("cerebrum.shutdown.start")
    await dispose_db_pool()
    await close_redis_pool()
    logger.info("cerebrum.shutdown.complete")


def create_application() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI application.
    Using a factory pattern allows clean testing by creating fresh instances.
    """
    configure_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)

    app = FastAPI(
        title="Cerebrum API",
        description="Enterprise Multi-Agent AI Platform — REST API",
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.APP_DEBUG else None,
        redoc_url="/redoc" if settings.APP_DEBUG else None,
        openapi_url="/openapi.json" if settings.APP_DEBUG else None,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        contact={
            "name": "Cerebrum Team",
            "email": "engineering@cerebrum.ai",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # ── Middleware (order matters: outermost first) ────────────────────────
    _configure_middleware(app)

    # ── Routers ───────────────────────────────────────────────────────────
    _configure_routers(app)

    # ── Prometheus Metrics ────────────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    return app


def _configure_middleware(app: FastAPI) -> None:
    """Register all middleware in correct order."""

    # Trusted hosts (security)
    if settings.APP_ENV == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-Process-Time"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Request ID (must be early in chain)
    app.add_middleware(RequestIDMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
        burst=settings.RATE_LIMIT_BURST,
    )

    # Audit logging
    app.add_middleware(AuditLogMiddleware)

    # Request timing
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        return response


def _configure_routers(app: FastAPI) -> None:
    """Register all API routers with their prefixes."""
    api_v1 = "/api/v1"

    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(auth.router, prefix=f"{api_v1}/auth", tags=["Authentication"])
    app.include_router(users.router, prefix=f"{api_v1}/users", tags=["Users"])
    app.include_router(projects.router, prefix=f"{api_v1}/projects", tags=["Projects"])
    app.include_router(datasets.router, prefix=f"{api_v1}/datasets", tags=["Datasets"])
    app.include_router(agents.router, prefix=f"{api_v1}/agents", tags=["Agents"])
    app.include_router(tasks.router, prefix=f"{api_v1}/tasks", tags=["Tasks"])
    app.include_router(ml.router, prefix=f"{api_v1}/ml", tags=["ML Pipeline"])
    app.include_router(
        visualizations.router,
        prefix=f"{api_v1}/visualizations",
        tags=["Visualizations"],
    )
    app.include_router(reports.router, prefix=f"{api_v1}/reports", tags=["Reports"])


# ── Application instance ────────────────────────────────────────────────────
app = create_application()
