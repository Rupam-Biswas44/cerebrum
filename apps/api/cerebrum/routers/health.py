"""
Health Check Router

Endpoints for liveness and readiness probes.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from cerebrum.config import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic liveness probe returning OK."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
