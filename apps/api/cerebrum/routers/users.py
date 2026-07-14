"""
Users Router

Endpoints for user registration, profile management, and RBAC-secured user listing.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.dependencies.auth import RequireAdmin, RequireAnyRole
from cerebrum.exceptions import ValidationError
from core.security.passwords import get_password_hash
from models.domain import User
from models.schemas.user import UserCreate, UserResponse

router = APIRouter()


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """Register a new user."""
    # Check if email exists
    stmt = select(User).where(User.email == user_in.email)
    existing_email = (await db.execute(stmt)).scalar_one_or_none()
    if existing_email:
        raise ValidationError("Email already registered")

    # Check if username exists
    stmt = select(User).where(User.username == user_in.username)
    existing_username = (await db.execute(stmt)).scalar_one_or_none()
    if existing_username:
        raise ValidationError("Username already taken")

    # Create new user
    db_user = User(
        email=user_in.email,
        username=user_in.username,
        full_name=user_in.full_name,
        role=user_in.role,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
    )

    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: RequireAnyRole,
) -> User:
    """Get current user's profile."""
    return current_user  # type: ignore[no-any-return]


@router.get("", response_model=list[UserResponse])
async def list_users(
    _admin: RequireAdmin,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    skip: int = 0,
    limit: int = 100,
) -> list[User]:
    """List all users (Admin only)."""
    stmt = select(User).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())
