"""
Cache Module — Async Redis Connection Pooling

This module handles the connection to Redis using the redis-py async client.
Provides connection pooling and a global Redis client for caching and rate limiting.
"""

from __future__ import annotations

import structlog
from redis.asyncio import Redis, from_url

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Global Redis client instance
redis_client: Redis | None = None


async def create_redis_pool() -> None:
    """
    Initialize the Redis connection pool.
    Called during the FastAPI lifespan startup event.
    """
    global redis_client

    if redis_client is not None:
        logger.warning("redis.pool.already_initialized")
        return

    logger.info("redis.pool.initializing")

    try:
        redis_client = from_url(
            str(settings.REDIS_URL),
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )
        
        # Test connection
        await redis_client.ping()

    except Exception as e:
        logger.error("redis.pool.failed", error=str(e))
        redis_client = None
        raise


async def close_redis_pool() -> None:
    """
    Close the Redis connection pool gracefully.
    Called during the FastAPI lifespan shutdown event.
    """
    global redis_client

    if redis_client is None:
        return

    logger.info("redis.pool.disposing")
    await redis_client.aclose()
    redis_client = None


def get_redis() -> Redis:
    """
    Get the global Redis client.
    Raises RuntimeError if the pool hasn't been initialized.
    """
    if redis_client is None:
        msg = "Redis pool not initialized"
        raise RuntimeError(msg)
    return redis_client
