"""
Visualizations Router

Endpoints for fetching generated visualizations.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from cerebrum.dependencies.auth import RequireAnyRole
from core.storage.minio import minio_client

router = APIRouter()


class VisualizationOut(BaseModel):
    artifact_url: str
    chart_spec: dict[str, Any]


@router.get(
    "/projects/{project_id}/tasks/{task_id}/visualizations", response_model=list[VisualizationOut]
)
async def list_visualizations(
    project_id: uuid.UUID,
    task_id: uuid.UUID,
    _current_user: RequireAnyRole,
) -> list[VisualizationOut]:
    """
    List all static visualizations generated for a specific task.
    """
    if minio_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service not configured.",
        )

    prefix = f"{project_id}/{task_id}/"
    try:
        objects = minio_client.list_objects("visualizations", prefix=prefix, recursive=True)
        results = []
        for obj in objects:
            if obj.object_name.endswith(".png"):
                results.append(
                    VisualizationOut(
                        artifact_url=f"/api/storage/visualizations/{obj.object_name}",
                        chart_spec={},  # In a real system, we'd fetch the JSON spec from DB
                    )
                )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list visualizations: {e}",
        ) from e
