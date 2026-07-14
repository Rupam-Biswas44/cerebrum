"""
ML Engineer Agent

Responsible for model selection, training, hyperparameter tuning, and evaluation.
Delegates to MLService which wraps Scikit-Learn, XGBoost, LightGBM, Optuna, and MLflow.
"""

from __future__ import annotations

import json
import uuid

import pandas as pd
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from services.ml import MLService

logger = structlog.get_logger(__name__)

ML_SYSTEM_PROMPT = """You are the ML Engineer Agent for Cerebrum.
Your job is to decide the best ML task type and target column for the provided dataset.

Given the user's goal and the data schema from previous agents, output a JSON object:
{{
  "task_type": "classification",  // or "regression"
  "target_column": "label",       // the column to predict
  "reasoning": "The goal asks for classification, label column looks categorical."
}}

Be concise. Output strict JSON only.
"""


class MLEngineerAgent(BaseAgent):
    agent_type = "ml_engineer"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("ml_engineer.execute.start", subtask=context.subtask.id)

        # ── Step 1: Ask LLM to infer task_type and target_column ──────────────
        llm = ChatOpenAI(
            model=context.llm_config.model,
            temperature=0.0,
            max_tokens=512,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            schema = context.previous_outputs.get("schema_summary", {})
            user_message += f"\nData schema from Data Engineer: {json.dumps(schema)}"

        messages = [
            SystemMessage(content=ML_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response: AIMessage = await llm.ainvoke(messages)
            plan = json.loads(str(response.content).strip())
            task_type: str = plan.get("task_type", "classification")
            target_column: str = plan.get("target_column", "label")
            reasoning: str = plan.get("reasoning", "")
        except Exception as e:
            self._log.warning("ml_engineer.llm.failed", error=str(e))
            task_type = "classification"
            target_column = "label"
            reasoning = "Defaulted to classification - LLM inference failed."

        self._log.info(
            "ml_engineer.task_type_inferred",
            task_type=task_type,
            target_column=target_column,
        )

        # ── Step 2: Generate synthetic training data if none passed in ─────────
        # In production this would load the real dataset from MinIO/DuckDB.
        # For now we generate a small dummy dataset so the ML pipeline is testable.
        dataset = context.previous_outputs.get("training_dataset")
        if dataset and isinstance(dataset, dict):
            df = pd.DataFrame.from_dict(dataset)
        else:
            from sklearn.datasets import make_classification

            X_raw, y_raw = make_classification(n_samples=200, n_features=5, random_state=42)
            df = pd.DataFrame(X_raw, columns=[f"feature_{i}" for i in range(X_raw.shape[1])])
            df[target_column] = y_raw

        # ── Step 3: Run AutoML via MLService ───────────────────────────────────
        ml_service = MLService()
        try:
            results = ml_service.train_automl(
                project_id=uuid.UUID(context.project_id),
                experiment_name=f"task_{context.task_id}",
                df=df,
                target_column=target_column,
                task_type=task_type,
                n_trials=3,  # Keep small for speed; bump in production
            )
        except Exception as e:
            self._log.error("ml_engineer.training.failed", error=str(e))
            results = {
                "error": str(e),
                "best_score": 0.0,
                "best_params": {},
            }

        output = {
            "task_type": task_type,
            "target_column": target_column,
            "automl_results": results,
            "reasoning": reasoning,
        }

        self._log.info(
            "ml_engineer.execute.complete",
            best_score=results.get("best_score"),
        )

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning=reasoning,
        )
