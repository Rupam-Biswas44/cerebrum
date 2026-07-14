"""Visualizations Router"""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def list_visualizations() -> list:
    return []
