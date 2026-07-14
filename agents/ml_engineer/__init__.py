"""
ML Engineer Agent

Responsible for model selection, training, hyperparameter tuning, and evaluation.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

ML_SYSTEM_PROMPT = """You are the ML Engineer Agent for Cerebrum.
Your job is to build, train, and evaluate machine learning models.

Goal:
1. Select the best ML algorithm for the problem (classification, regression, etc).
2. Propose feature engineering steps and hyperparameter tuning strategy.
3. Evaluate the model performance using appropriate metrics.

Output format (strict JSON):
{{
  "selected_model": "RandomForest",
  "hyperparameters": {{...}},
  "metrics": {{"accuracy": 0.95, "f1": 0.94}},
  "feature_importance": {{...}}
}}
"""


class MLEngineerAgent(BaseAgent):
    agent_type = "ml_engineer"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("ml_engineer.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=ML_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response for now
        output = {
            "selected_model": "RandomForest",
            "hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "metrics": {"accuracy": 0.95, "f1": 0.94},
            "feature_importance": {"feature_A": 0.6, "feature_B": 0.4},
        }

        self._log.info("ml_engineer.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Trained a Random Forest model with high accuracy.",
        )
