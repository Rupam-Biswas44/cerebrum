"""
Datasets Router

Endpoints for uploading and managing datasets.
Supports CSV, JSON, Parquet, and Excel formats.
"""

import io
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.dependencies.auth import RequireAnyRole
from cerebrum.exceptions import ValidationError
from core.storage.minio import upload_file_stream
from models.domain import Dataset
from models.schemas.dataset import DatasetResponse
from services.data_profiler import profile_dataset

logger = structlog.get_logger(__name__)
router = APIRouter()

# Max file size currently configured for API: 50MB for synchronous profiling
MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    project_id: Annotated[uuid.UUID, Form(...)],
    file: UploadFile = File(...),
    description: str | None = Form(None),
) -> Dataset:
    """
    Upload a new dataset to a project.
    Automatically profiles the dataset to infer schema and statistics.
    """
    if not file.filename:
        raise ValidationError("Filename is missing")

    # Read file into memory (in a production environment, extremely large files
    # should be streamed directly to MinIO and profiled via async celery workers).
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE:
        raise ValidationError(
            f"File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit for synchronous upload."
        )

    # 1. Profile the dataset (schema inference, null count, etc.) using Polars
    try:
        profile = profile_dataset(
            file_data=file_bytes,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        raise ValidationError(f"Data profiling failed: {e}") from e

    # 2. Upload raw file to MinIO Object Storage
    bucket_name = f"project-{project_id}"
    object_name = f"datasets/{uuid.uuid4()}-{file.filename}"

    try:
        file_stream = io.BytesIO(file_bytes)
        file_path = upload_file_stream(
            bucket_name=bucket_name,
            object_name=object_name,
            data=file_stream,
            size=file_size,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logger.error("dataset.upload.minio_error", error=str(e))
        raise ValidationError("Failed to store file in object storage.") from e

    # 3. Save metadata to PostgreSQL
    db_dataset = Dataset(
        project_id=project_id,
        name=file.filename,
        description=description,
        file_path=file_path,
        file_size_bytes=file_size,
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        schema_metadata=profile["schema"],
    )

    db.add(db_dataset)
    await db.commit()
    await db.refresh(db_dataset)

    logger.info(
        "dataset.uploaded",
        dataset_id=str(db_dataset.id),
        project_id=str(project_id),
        user_id=str(current_user.id),
    )

    return db_dataset


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    project_id: uuid.UUID,
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[Dataset]:
    """List all datasets in a project."""
    # Note: In a real app we'd verify the user has access to `project_id`
    stmt = select(Dataset).where(Dataset.project_id == project_id, Dataset.deleted_at.is_(None))
    result = await db.execute(stmt)
    return list(result.scalars().all())
