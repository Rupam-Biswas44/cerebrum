"""
Vector Store Service (Qdrant)

Handles all interactions with Qdrant — the vector database that powers
semantic search, long-term agent memory, and knowledge retrieval.

Architecture:
  - Embeddings are generated via SentenceTransformer (local, no API cost).
  - Vectors are stored in Qdrant collections namespaced by project.
  - Metadata (payload) stored alongside vectors for rich filtering.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Embedding dimension for all-MiniLM-L6-v2 (fast, high quality)
EMBEDDING_DIM = 384
COLLECTION_PREFIX = "cerebrum"


def get_qdrant_client() -> AsyncQdrantClient:
    """Return an async Qdrant client connected to the configured host."""
    return AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=30,
    )


def _get_embedder():  # noqa: ANN202
    """Lazy-load the SentenceTransformer model to avoid import overhead."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


_embedder = None


def get_embedder():  # noqa: ANN202
    """Singleton accessor for the embedding model."""
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        logger.info("embedding_model.loading", model="all-MiniLM-L6-v2")
        _embedder = _get_embedder()
        logger.info("embedding_model.loaded")
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate dense vector embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors (each of length EMBEDDING_DIM).
    """
    embedder = get_embedder()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vec.tolist() for vec in embeddings]


def _collection_name(project_id: uuid.UUID) -> str:
    """Construct the Qdrant collection name for a given project."""
    return f"{COLLECTION_PREFIX}_{str(project_id).replace('-', '_')}"


async def ensure_collection(client: AsyncQdrantClient, project_id: uuid.UUID) -> str:
    """
    Ensure a Qdrant collection exists for the project. Creates it if missing.

    Returns the collection name.
    """
    coll = _collection_name(project_id)
    existing = [c.name for c in (await client.get_collections()).collections]

    if coll not in existing:
        await client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("qdrant.collection.created", collection=coll)

    return coll


class VectorMemory(BaseModel):
    """A single piece of content stored as a vector in Qdrant."""

    id: str
    text: str
    payload: dict[str, Any]


async def upsert_vectors(
    project_id: uuid.UUID,
    memories: list[VectorMemory],
) -> None:
    """
    Embed and upsert a list of text memories into the project's Qdrant collection.

    Args:
        project_id: The project namespace.
        memories: List of VectorMemory objects to index.
    """
    if not memories:
        return

    client = get_qdrant_client()
    collection = await ensure_collection(client, project_id)

    texts = [m.text for m in memories]
    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=m.id,
            vector=vec,
            payload={**m.payload, "text": m.text},
        )
        for m, vec in zip(memories, vectors, strict=True)
    ]

    await client.upsert(collection_name=collection, points=points)
    logger.info("qdrant.upsert.complete", collection=collection, count=len(points))
    await client.close()


async def semantic_search(
    project_id: uuid.UUID,
    query: str,
    top_k: int = 5,
    filter_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform cosine-similarity vector search against the project's collection.

    Args:
        project_id: Project namespace to search in.
        query: Natural language query string.
        top_k: Number of top results to return.
        filter_payload: Optional Qdrant payload filter (e.g., {"memory_type": "insight"}).

    Returns:
        List of result dicts with 'score', 'text', and 'payload' fields.
    """
    client = get_qdrant_client()
    collection = await ensure_collection(client, project_id)

    query_vector = embed_texts([query])[0]

    qdrant_filter = None
    if filter_payload:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_payload.items()
            ]
        )

    results = await client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    await client.close()

    return [
        {
            "score": r.score,
            "text": r.payload.get("text", ""),
            "payload": {k: v for k, v in r.payload.items() if k != "text"},
            "id": str(r.id),
        }
        for r in results
    ]
