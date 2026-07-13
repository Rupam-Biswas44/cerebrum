"""
Auth Schemas

Pydantic models for authentication requests and responses.
"""

from typing import Any
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    role: str
    type: str
    exp: int


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
