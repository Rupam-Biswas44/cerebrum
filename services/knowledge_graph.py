"""
Neo4j Knowledge Graph Service

Stores and queries entity relationships in a graph database.
Used by agents to:
  - Map relationships between datasets, models, tasks, and users.
  - Trace data lineage (which model was trained on which dataset).
  - Build contextual understanding graphs from ingested documents.

Connection is managed as an application-level singleton.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """Return the singleton Neo4j async driver, initialising it if needed."""
    global _driver  # noqa: PLW0603
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("neo4j.driver.initialised", uri=settings.NEO4J_URL)
    return _driver


async def close_neo4j_driver() -> None:
    """Close the driver — called on application shutdown."""
    global _driver  # noqa: PLW0603
    if _driver:
        await _driver.close()
        _driver = None
        logger.info("neo4j.driver.closed")


# ============================================================
# Graph Schema Bootstrap
# ============================================================


async def ensure_constraints() -> None:
    """
    Ensure database constraints and indexes exist.
    Safe to run on every startup (CREATE CONSTRAINT IF NOT EXISTS).
    """
    driver = get_neo4j_driver()
    async with driver.session() as session:
        queries = [
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:MLModel) REQUIRE m.id IS UNIQUE",
        ]
        for q in queries:
            await session.run(q)
    logger.info("neo4j.constraints.ensured")


# ============================================================
# Node Operations
# ============================================================


async def upsert_project_node(project_id: uuid.UUID, name: str, owner_id: uuid.UUID) -> None:
    """Create or update a Project node and link it to its owner User."""
    driver = get_neo4j_driver()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (u:User {id: $owner_id})
            MERGE (p:Project {id: $project_id})
            SET p.name = $name, p.updated_at = datetime()
            MERGE (u)-[:OWNS]->(p)
            """,
            project_id=str(project_id),
            name=name,
            owner_id=str(owner_id),
        )
    logger.info("neo4j.project.upserted", project_id=str(project_id))


async def upsert_dataset_node(
    dataset_id: uuid.UUID,
    project_id: uuid.UUID,
    name: str,
    row_count: int | None = None,
    column_count: int | None = None,
) -> None:
    """Create or update a Dataset node and link it to its Project."""
    driver = get_neo4j_driver()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (p:Project {id: $project_id})
            MERGE (d:Dataset {id: $dataset_id})
            SET d.name = $name, d.row_count = $row_count, d.column_count = $column_count
            MERGE (p)-[:CONTAINS]->(d)
            """,
            dataset_id=str(dataset_id),
            project_id=str(project_id),
            name=name,
            row_count=row_count,
            column_count=column_count,
        )
    logger.info("neo4j.dataset.upserted", dataset_id=str(dataset_id))


async def link_model_to_dataset(  # noqa: E501
    model_id: uuid.UUID,
    dataset_id: uuid.UUID,
    experiment_id: uuid.UUID,
) -> None:
    """Record that an ML model was trained on a specific dataset (data lineage)."""
    driver = get_neo4j_driver()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (d:Dataset {id: $dataset_id})
            MERGE (m:MLModel {id: $model_id})
            SET m.experiment_id = $experiment_id
            MERGE (m)-[:TRAINED_ON]->(d)
            """,
            model_id=str(model_id),
            dataset_id=str(dataset_id),
            experiment_id=str(experiment_id),
        )
    logger.info("neo4j.model_dataset.linked", model_id=str(model_id), dataset_id=str(dataset_id))


# ============================================================
# Graph Queries
# ============================================================


async def get_project_lineage(project_id: uuid.UUID) -> list[dict[str, Any]]:
    """
    Return the full data lineage graph for a project:
    which datasets were used to train which models.
    """
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(d:Dataset)
            OPTIONAL MATCH (m:MLModel)-[:TRAINED_ON]->(d)
            RETURN d.id AS dataset_id, d.name AS dataset_name,
                   collect(m.id) AS model_ids
            """,
            project_id=str(project_id),
        )
        records = await result.data()

    return [
        {
            "dataset_id": r["dataset_id"],
            "dataset_name": r["dataset_name"],
            "model_ids": r["model_ids"],
        }
        for r in records
    ]


async def find_related_entities(entity_id: str, depth: int = 2) -> list[dict[str, Any]]:
    """
    Traverse the knowledge graph up to `depth` hops from a given node.
    Useful for agents to understand context around any entity.
    """
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run(
            f"""
            MATCH path = (start {{id: $entity_id}})-[*1..{depth}]-(end)
            RETURN DISTINCT
                labels(end)[0] AS entity_type,
                end.id AS entity_id,
                end.name AS entity_name,
                length(path) AS hops
            ORDER BY hops ASC
            LIMIT 50
            """,
            entity_id=entity_id,
        )
        records = await result.data()

    return [
        {
            "entity_type": r["entity_type"],
            "entity_id": r["entity_id"],
            "entity_name": r["entity_name"],
            "hops": r["hops"],
        }
        for r in records
    ]
