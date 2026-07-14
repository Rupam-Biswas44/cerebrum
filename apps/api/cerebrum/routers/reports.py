"""
Reports Router

Endpoints for fetching generated reports (Markdown, PDF, PPTX).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from cerebrum.dependencies.auth import RequireAnyRole
from core.storage.minio import minio_client

router = APIRouter()


class ReportOut(BaseModel):
    artifact_url: str
    format: str


@router.get("/projects/{project_id}/tasks/{task_id}/reports", response_model=list[ReportOut])
async def list_reports(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _current_user: RequireAnyRole,
) -> list[ReportOut]:
    """
    List all reports generated for a specific task.
    """
    if minio_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service not configured.",
        )

    prefix = f"{project_id}/{task_id}/"
    try:
        objects = minio_client.list_objects("reports", prefix=prefix, recursive=True)
        results = []
        for obj in objects:
            ext = obj.object_name.split(".")[-1] if "." in obj.object_name else "unknown"
            results.append(
                ReportOut(
                    artifact_url=f"/api/storage/reports/{obj.object_name}",
                    format=ext,
                )
            )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {e}",
        ) from e
