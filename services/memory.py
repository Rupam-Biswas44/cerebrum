"""
Memory Service — 3-Tier Agent Memory System

Cerebrum implements three tiers of memory for AI agents:

  Tier 1 — Short-Term (Redis TTL cache):
      Fast, ephemeral working memory for the current task session.
      Disappears after a configurable TTL (e.g. 1 hour).

  Tier 2 — Medium-Term (PostgreSQL):
      Persistent structured memory per user/project. Survives sessions.
      Used to recall past tasks, decisions, and user preferences.

  Tier 3 — Long-Term (Qdrant vector store + Neo4j knowledge graph):
      Semantic memory. Agent stores compressed summaries as dense vectors
      so it can retrieve relevant context by meaning, not just keywords.
      Neo4j stores relationship graphs between entities (e.g., who is the
      owner of which project, how datasets relate to models).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# Tier 1 — Short-Term Memory (Redis)
# ============================================================

async def store_short_term(
    redis: Any,
    session_id: str,
    key: str,
    value: Any,
    ttl_seconds: int = 3600,
) -> None:
    """
    Store a value in Redis for the duration of a task session.

    Args:
        redis: The active aioredis client.
        session_id: The current agent task session ID.
        key: A named key for this piece of memory (e.g. 'last_query').
        value: Any JSON-serialisable object.
        ttl_seconds: Expiry time in seconds (default 1 hour).
    """
    redis_key = f"memory:short:{session_id}:{key}"
    serialized = json.dumps(value, default=str)
    await redis.setex(redis_key, ttl_seconds, serialized)
    logger.debug("memory.short_term.stored", key=redis_key, ttl=ttl_seconds)


async def get_short_term(
    redis: Any,
    session_id: str,
    key: str,
) -> Any | None:
    """Retrieve a value from the Redis short-term memory cache."""
    redis_key = f"memory:short:{session_id}:{key}"
    raw = await redis.get(redis_key)
    if raw is None:
        return None
    return json.loads(raw)


async def clear_short_term_session(redis: Any, session_id: str) -> None:
    """Delete all short-term memory keys for a given session."""
    pattern = f"memory:short:{session_id}:*"
    keys = await redis.keys(pattern)
    if keys:
        await redis.delete(*keys)
    logger.info("memory.short_term.session_cleared", session_id=session_id, keys_deleted=len(keys))


# ============================================================
# Tier 2 — Medium-Term Memory (PostgreSQL AgentMemory records)
# ============================================================

async def store_medium_term(
    db: Any,
    user_id: uuid.UUID,
    project_id: uuid.UUID | None,
    memory_type: str,
    content: dict[str, Any],
    importance: float = 0.5,
    expires_at: datetime | None = None,
) -> None:
    """
    Persist a structured memory record to PostgreSQL.

    Args:
        db: Active AsyncSession.
        user_id: Owner of this memory.
        project_id: Optional project scope.
        memory_type: Type tag e.g. 'task_summary', 'user_preference', 'insight'.
        content: The data payload (JSON).
        importance: Relevance score 0.0–1.0 (used for retrieval ranking).
        expires_at: Optional expiry timestamp for auto-pruning.
    """
    from sqlalchemy import text

    stmt = text("""
        INSERT INTO agent_memories
            (id, user_id, project_id, memory_type, content, importance, expires_at, created_at, updated_at)
        VALUES
            (:id, :user_id, :project_id, :memory_type, :content, :importance, :expires_at, now(), now())
        ON CONFLICT DO NOTHING
    """)

    await db.execute(stmt, {
        "id": uuid.uuid4(),
        "user_id": str(user_id),
        "project_id": str(project_id) if project_id else None,
        "memory_type": memory_type,
        "content": json.dumps(content),
        "importance": importance,
        "expires_at": expires_at,
    })
    await db.commit()
    logger.info("memory.medium_term.stored", user_id=str(user_id), memory_type=memory_type)


async def get_medium_term(
    db: Any,
    user_id: uuid.UUID,
    memory_type: str | None = None,
    project_id: uuid.UUID | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Retrieve recent medium-term memory records from PostgreSQL.
    Results are sorted by importance × recency.
    """
    from sqlalchemy import text

    filters = ["user_id = :user_id", "(expires_at IS NULL OR expires_at > now())"]
    params: dict[str, Any] = {"user_id": str(user_id), "limit": limit}

    if memory_type:
        filters.append("memory_type = :memory_type")
        params["memory_type"] = memory_type

    if project_id:
        filters.append("project_id = :project_id")
        params["project_id"] = str(project_id)

    where_clause = " AND ".join(filters)
    stmt = text(f"""
        SELECT id, memory_type, content, importance, created_at
        FROM agent_memories
        WHERE {where_clause}
        ORDER BY importance DESC, created_at DESC
        LIMIT :limit
    """)

    result = await db.execute(stmt, params)
    rows = result.fetchall()

    return [
        {
            "id": str(row.id),
            "memory_type": row.memory_type,
            "content": json.loads(row.content),
            "importance": row.importance,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
