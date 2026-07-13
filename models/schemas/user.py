"""
User Schemas

Pydantic models for User creation, reading, and updating.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from models.domain import UserRole


class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    full_name: str | None = Field(default=None, max_length=200)
    avatar_url: str | None = None
    role: UserRole = UserRole.ANALYST
    preferences: dict[str, Any] = Field(default_factory=dict)


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    username: str | None = Field(default=None, min_length=3, max_length=50)
    full_name: str | None = Field(default=None, max_length=200)
    avatar_url: str | None = None
    preferences: dict[str, Any] | None = None


class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
