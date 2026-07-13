"""ML Router"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/experiments")
async def list_experiments() -> list:
    return []
