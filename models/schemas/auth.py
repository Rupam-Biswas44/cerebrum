"""
Auth Schemas

Pydantic models for authentication requests and responses.
"""

from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105


class TokenPayload(BaseModel):
    sub: str
    role: str
    type: str
    exp: int


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
