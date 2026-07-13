"""
JWT Generation and Validation

Provides utilities for creating and decoding JSON Web Tokens (JWT)
for access and refresh token flows.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt, JWTError
from cerebrum.config import get_settings

settings = get_settings()


def create_access_token(
    subject: str | Any,
    role: str = "analyst",
    expires_delta: timedelta | None = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: Typically the user_id as a string.
        role: The user's role (admin, analyst, viewer).
        expires_delta: Optional custom expiration timedelta.
        
    Returns:
        The encoded JWT token as a string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "role": role,
        "type": "access",
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def create_refresh_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT refresh token. (Typically has a longer lifespan).
    
    Args:
        subject: Typically the user_id as a string.
        expires_delta: Optional custom expiration timedelta.
        
    Returns:
        The encoded JWT token as a string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh",
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The encoded JWT string.
        
    Returns:
        The decoded payload as a dictionary.
        
    Raises:
        JWTError: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError:
        raise
