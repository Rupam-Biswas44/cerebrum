"""Users Router"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/me")
async def read_users_me() -> dict:
    return {"user_id": "dummy"}
