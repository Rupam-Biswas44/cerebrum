"""
Tasks Router

Endpoints to create, run, and monitor multi-agent AI tasks.
A Task represents a full orchestrated pipeline run.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cerebrum.core.database import get_db_session
from cerebrum.dependencies.auth import RequireAnyRole
from models.domain import Task, TaskStatus

logger = structlog.get_logger(__name__)
router = APIRouter()


# ============================================================
# Request / Response Schemas
# ============================================================


class CreateTaskRequest(BaseModel):
    project_id: uuid.UUID
    goal: str = Field(..., min_length=10, description="Natural language goal for the AI agents")
    task_type: Literal["analysis", "ml_training", "report", "custom"] = "analysis"


class TaskResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    goal: str
    status: str
    task_type: str | None = None
    final_output: dict[str, Any]
    total_tokens: int
    total_cost_usd: float
    error: str | None = None

    class Config:
        from_attributes = True


# ============================================================
# Background Runner
# ============================================================


async def _run_task_in_background(
    task_id: str,
    project_id: str,
    user_id: str,
    goal: str,
    task_type: str,
) -> None:
    """
    Runs the multi-agent orchestrator in a FastAPI BackgroundTask.
    Updates the Task record in PostgreSQL with the final result.
    """
    from sqlalchemy import update

    from agents.orchestrator import run_orchestrator
    from cerebrum.core.database import get_db_pool

    logger.info("task.background.start", task_id=task_id)

    try:
        # Run the full LangGraph orchestration pipeline
        final_state = await run_orchestrator(
            task_id=task_id,
            project_id=project_id,
            user_id=user_id,
            goal=goal,
            task_type=task_type,
        )

        status = TaskStatus.COMPLETED if not final_state.get("error") else TaskStatus.FAILED

        # Persist result back to DB
        async with get_db_pool().begin() as conn:
            from sqlalchemy.ext.asyncio import AsyncSession as AsyncSess

            db = AsyncSess(bind=conn)
            await db.execute(
                update(Task)
                .where(Task.id == uuid.UUID(task_id))
                .values(
                    status=status,
                    final_output=final_state.get("agent_outputs", {}),
                    error=final_state.get("error"),
                )
            )

    except Exception as e:
        logger.error("task.background.failed", task_id=task_id, error=str(e))


# ============================================================
# Endpoints
# ============================================================


@router.post("", response_model=TaskResponse, status_code=202)
async def create_task(
    body: CreateTaskRequest,
    current_user: RequireAnyRole,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> Task:
    """
    Create and asynchronously execute a multi-agent AI task.

    The task is immediately saved to PostgreSQL with status=PENDING,
    the HTTP response is returned (202 Accepted), and the full
    orchestration pipeline runs in the background.
    """
    db_task = Task(
        project_id=body.project_id,
        created_by=current_user.id,
        goal=body.goal,
        status=TaskStatus.PENDING,
        plan={},
        final_output={},
    )
    db.add(db_task)
    await db.commit()
    await db.refresh(db_task)

    # Kick off the orchestration in the background (non-blocking)
    background_tasks.add_task(
        _run_task_in_background,
        task_id=str(db_task.id),
        project_id=str(body.project_id),
        user_id=str(current_user.id),
        goal=body.goal,
        task_type=body.task_type,
    )

    logger.info(
        "task.created",
        task_id=str(db_task.id),
        goal=body.goal[:80],
        user_id=str(current_user.id),
    )

    return db_task


@router.get("", response_model=list[TaskResponse])
async def list_tasks(
    project_id: uuid.UUID,
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[Task]:
    """List all tasks for a project."""
    stmt = select(Task).where(Task.project_id == project_id).order_by(Task.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    current_user: RequireAnyRole,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> Task:
    """Get a specific task and its current status / outputs."""
    from cerebrum.exceptions import NotFoundError

    stmt = select(Task).where(Task.id == task_id)
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    if not task:
        raise NotFoundError("Task", str(task_id))
    return task
