"""
Auth Dependencies

FastAPI dependencies for injecting the current authenticated user
and enforcing Role-Based Access Control (RBAC).
"""

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.exceptions import AuthenticationError, AuthorizationError
from core.security.jwt import decode_token
from models.domain import User, UserRole

# We define the token URL for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    request: Request,
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """
    Dependency that decodes the JWT token and fetches the User from DB.
    Raises AuthenticationError if token is invalid or user is missing/inactive.
    """
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        token_type = payload.get("type")

        if user_id is None:
            raise AuthenticationError("Could not validate credentials")
        if token_type != "access":
            raise AuthenticationError("Invalid token type")

    except JWTError as e:
        raise AuthenticationError(f"Token validation failed: {e}") from e

    # Fetch user from database
    stmt = select(User).where(User.id == user_id, User.deleted_at.is_(None))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise AuthenticationError("User not found")
    if not user.is_active:
        raise AuthenticationError("User account is inactive")

    # Attach user_id to request state for middleware (e.g. RateLimit, AuditLog)
    request.state.user_id = user.id

    return user


def require_role(allowed_roles: list[UserRole]):
    """
    Dependency factory to enforce RBAC.
    Usage: Depends(require_role([UserRole.ADMIN, UserRole.ANALYST]))
    """

    async def role_checker(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        if current_user.role not in allowed_roles:
            raise AuthorizationError(
                f"Role '{current_user.role}' lacks required permissions. "
                f"Allowed roles: {[r.value for r in allowed_roles]}"
            )
        return current_user

    return role_checker


# Convenience dependencies
RequireAdmin = Annotated[User, Depends(require_role([UserRole.ADMIN]))]
RequireAnalyst = Annotated[User, Depends(require_role([UserRole.ADMIN, UserRole.ANALYST]))]
RequireAnyRole = Annotated[User, Depends(get_current_user)]
