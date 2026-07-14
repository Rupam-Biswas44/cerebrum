"""
ML Router

Endpoints for querying MLflow experiments, retrieving model metrics,
listing runs, and triggering ad-hoc AutoML jobs.
"""

from __future__ import annotations

import uuid
from typing import Any

import mlflow
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from cerebrum.dependencies.auth import RequireAnyRole

router = APIRouter()


# ============================================================
# Schemas
# ============================================================


class ExperimentOut(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str | None = None
    lifecycle_stage: str


class RunMetrics(BaseModel):
    run_id: str
    run_name: str | None = None
    status: str
    metrics: dict[str, float]
    params: dict[str, str]
    start_time: int | None = None
    end_time: int | None = None


class AutoMLRequest(BaseModel):
    project_id: uuid.UUID
    experiment_name: str = Field(..., min_length=3)
    target_column: str = Field(..., min_length=1)
    task_type: str = Field("classification", pattern="^(classification|regression)$")
    n_trials: int = Field(5, ge=1, le=50)


# ============================================================
# Endpoints
# ============================================================


@router.get("/experiments", response_model=list[ExperimentOut])
async def list_experiments(
    _current_user: RequireAnyRole,
) -> list[ExperimentOut]:
    """
    List all MLflow experiments.
    """
    try:
        experiments = mlflow.search_experiments()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow unavailable: {e}",
        ) from e

    return [
        ExperimentOut(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
        )
        for exp in experiments
    ]


@router.get("/experiments/{experiment_id}/runs", response_model=list[RunMetrics])
async def list_runs(
    experiment_id: str,
    _current_user: RequireAnyRole,
) -> list[RunMetrics]:
    """
    List all runs in an MLflow experiment.
    """
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            output_format="list",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow unavailable: {e}",
        ) from e

    result: list[RunMetrics] = []
    for run in runs:
        info = run.info
        result.append(
            RunMetrics(
                run_id=info.run_id,
                run_name=info.run_name,
                status=info.status,
                metrics=dict(run.data.metrics.items()),
                params=dict(run.data.params.items()),
                start_time=info.start_time,
                end_time=info.end_time,
            )
        )

    return result


@router.get("/experiments/{experiment_id}/best-run", response_model=RunMetrics)
async def get_best_run(
    experiment_id: str,
    metric: str = "val_f1",
    _current_user: RequireAnyRole = None,  # type: ignore[assignment]
) -> RunMetrics:
    """
    Retrieve the best run in an experiment by a given metric.
    """
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
            output_format="list",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow unavailable: {e}",
        ) from e

    if not runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No runs found in experiment {experiment_id}",
        )

    run = runs[0]
    info = run.info
    return RunMetrics(
        run_id=info.run_id,
        run_name=info.run_name,
        status=info.status,
        metrics=dict(run.data.metrics.items()),
        params=dict(run.data.params.items()),
        start_time=info.start_time,
        end_time=info.end_time,
    )


@router.post("/automl", status_code=status.HTTP_202_ACCEPTED)
async def trigger_automl(
    request: AutoMLRequest,
    current_user: RequireAnyRole,
) -> dict[str, Any]:
    """
    Trigger an AutoML run in the background via a direct MLService call.
    Returns the run ID immediately while the job processes asynchronously.
    Note: For production, this should dispatch to a Celery worker.
    """
    import asyncio

    import pandas as pd
    from sklearn.datasets import make_classification, make_regression

    from services.ml import MLService

    # Create synthetic dataset for demonstration
    if request.task_type == "classification":
        X_raw, y_raw = make_classification(n_samples=300, n_features=8, random_state=42)
    else:
        X_raw, y_raw = make_regression(n_samples=300, n_features=8, random_state=42)

    df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])
    df[request.target_column] = y_raw

    ml_service = MLService()
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ml_service.train_automl(
                project_id=request.project_id,
                experiment_name=request.experiment_name,
                df=df,
                target_column=request.target_column,
                task_type=request.task_type,
                n_trials=request.n_trials,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AutoML failed: {e}",
        ) from e

    return {
        "status": "completed",
        "project_id": str(request.project_id),
        "experiment_name": request.experiment_name,
        "results": results,
    }
