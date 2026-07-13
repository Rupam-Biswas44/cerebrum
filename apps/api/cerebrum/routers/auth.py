"""Auth Router"""
from fastapi import APIRouter
router = APIRouter()

@router.post("/token")
async def login() -> dict:
    return {"access_token": "dummy", "token_type": "bearer"}
