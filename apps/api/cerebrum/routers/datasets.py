"""Datasets Router"""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_datasets() -> list:
    return []
