"""
Database Module — Async SQLAlchemy Connection Pooling

This module handles the application's connection to PostgreSQL using
asyncpg and SQLAlchemy 2.x. It sets up connection pooling and provides
the async session maker for dependency injection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Global engine and sessionmaker instances
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def create_db_pool() -> None:
    """
    Initialize the database connection pool.
    Called during the FastAPI lifespan startup event.
    """
    global engine, async_session_factory

    if engine is not None:
        logger.warning("database.pool.already_initialized")
        return

    logger.info(
        "database.pool.initializing",
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
    )

    try:
        engine = create_async_engine(
            str(settings.DATABASE_URL),
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_pre_ping=settings.DATABASE_POOL_PRE_PING,
            echo=settings.APP_DEBUG,  # Log SQL queries if debug is True
        )

        async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    except Exception as e:
        logger.error("database.pool.failed", error=str(e))
        raise


async def dispose_db_pool() -> None:
    """
    Dispose of the database connection pool gracefully.
    Called during the FastAPI lifespan shutdown event.
    """
    global engine, async_session_factory

    if engine is None:
        return

    logger.info("database.pool.disposing")
    await engine.dispose()
    engine = None
    async_session_factory = None


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """
    FastAPI dependency that provides a transactional database session.
    Automatically commits on success or rolls back on exception.
    """
    if async_session_factory is None:
        msg = "Database pool not initialized"
        raise RuntimeError(msg)

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
