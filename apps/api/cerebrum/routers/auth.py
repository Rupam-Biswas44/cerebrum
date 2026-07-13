"""
Auth Router

Endpoints for obtaining and refreshing JWT tokens.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.exceptions import AuthenticationError
from core.security.jwt import create_access_token, create_refresh_token, decode_token
from core.security.passwords import verify_password
from models.domain import User
from models.schemas.auth import Token

router = APIRouter()


@router.post("/token", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> Token:
    """
    OAuth2 compatible token login. 
    Accepts application/x-www-form-urlencoded with 'username' and 'password'.
    (In this application, 'username' maps to the user's email).
    """
    # Fetch user by email
    stmt = select(User).where(User.email == form_data.username, User.deleted_at.is_(None))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError("Incorrect email or password")
    
    if not user.hashed_password:
        raise AuthenticationError("User registered via OAuth, cannot login with password")

    if not verify_password(form_data.password, user.hashed_password):
        raise AuthenticationError("Incorrect email or password")
        
    if not user.is_active:
        raise AuthenticationError("Account is inactive")

    # Generate tokens
    access_token = create_access_token(subject=str(user.id), role=user.role.value)
    refresh_token = create_refresh_token(subject=str(user.id))

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> Token:
    """Issue a new access token using a valid refresh token."""
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type. Refresh token required.")
        user_id = payload.get("sub")
    except Exception as e:
        raise AuthenticationError(f"Invalid refresh token: {e}")

    stmt = select(User).where(User.id == user_id, User.deleted_at.is_(None))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")

    # Issue a new access token only
    access_token = create_access_token(subject=str(user.id), role=user.role.value)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )

