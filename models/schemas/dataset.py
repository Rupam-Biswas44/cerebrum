"""
Dataset Schemas

Pydantic models for Dataset upload responses and metadata listing.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ColumnProfile(BaseModel):
    name: str
    type: str
    null_count: int
    null_percentage: float
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    schema: list[ColumnProfile]


class DatasetResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None = None
    file_path: str
    file_size_bytes: int
    row_count: int | None = None
    column_count: int | None = None
    schema_metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
