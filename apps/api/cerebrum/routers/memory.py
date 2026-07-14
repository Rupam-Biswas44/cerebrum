"""
Memory Retrieval API Router

Exposes agent memory APIs for reading, writing, and searching
across the 3-tier memory system (Redis, PostgreSQL, Qdrant).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.dependencies.auth import RequireAnyRole
from services.memory import get_medium_term, store_medium_term
from services.vector_store import VectorMemory, semantic_search, upsert_vectors

router = APIRouter()


# ============================================================
# Request / Response Models
# ============================================================


class StoreMemoryRequest(BaseModel):
    memory_type: str = Field(  # noqa: E501
        ..., description="Type tag e.g. 'insight', 'preference', 'task_summary'"
    )
    content: dict[str, Any] = Field(..., description="Arbitrary JSON content to store")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    project_id: uuid.UUID | None = None
    index_for_search: bool = Field(
        default=True, description="If True, also embed and store in Qdrant for semantic search"
    )


class MemoryRecord(BaseModel):
    id: str
    memory_type: str
    content: dict[str, Any]
    importance: float
    created_at: str


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    project_id: uuid.UUID
    top_k: int = Field(default=5, ge=1, le=20)
    filter_type: str | None = None


class SemanticSearchResult(BaseModel):
    score: float
    text: str
    payload: dict[str, Any]
    id: str


# ============================================================
# Endpoints
# ============================================================


@router.post("", status_code=201)
async def store_memory(
    body: StoreMemoryRequest,
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, str]:
    """
    Store a memory record in PostgreSQL (medium-term).
    Optionally also index in Qdrant for semantic retrieval.
    """
    await store_medium_term(
        db=db,
        user_id=current_user.id,
        project_id=body.project_id,
        memory_type=body.memory_type,
        content=body.content,
        importance=body.importance,
    )

    # If the content has a 'text' field, embed it for vector search
    if body.index_for_search and body.project_id and "text" in body.content:
        mem = VectorMemory(
            id=str(uuid.uuid4()),
            text=body.content["text"],
            payload={
                "memory_type": body.memory_type,
                "user_id": str(current_user.id),
                "importance": body.importance,
            },
        )
        await upsert_vectors(project_id=body.project_id, memories=[mem])

    return {"status": "stored"}


@router.get("", response_model=list[MemoryRecord])
async def list_memories(
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    memory_type: str | None = Query(default=None),
    project_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """List recent medium-term memories for the current user."""
    return await get_medium_term(
        db=db,
        user_id=current_user.id,
        memory_type=memory_type,
        project_id=project_id,
        limit=limit,
    )


@router.post("/search", response_model=list[SemanticSearchResult])
async def search_memories(
    body: SemanticSearchRequest,
    current_user: RequireAnyRole,
) -> list[dict[str, Any]]:
    """
    Perform semantic vector search over long-term memory using Qdrant.
    Returns the top-k most relevant memories ranked by cosine similarity.
    """
    filter_payload = None
    if body.filter_type:
        filter_payload = {"memory_type": body.filter_type}

    results = await semantic_search(
        project_id=body.project_id,
        query=body.query,
        top_k=body.top_k,
        filter_payload=filter_payload,
    )
    return results
