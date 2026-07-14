"""
Statistician Agent

Responsible for Exploratory Data Analysis (EDA), hypothesis testing, and correlation analysis.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

STAT_SYSTEM_PROMPT = """You are the Statistician Agent for Cerebrum.
Your job is to perform statistical analysis, hypothesis testing, and EDA on data
prepared by the Data Engineer.

Goal:
1. Analyze distributions and summary statistics.
2. Identify correlations and statistical significance.
3. Return a JSON object with your findings.

Output format (strict JSON):
{{
  "key_findings": ["finding 1", "finding 2"],
  "correlations": {{...}},
  "statistical_tests": [{{...}}]
}}
"""


class StatisticianAgent(BaseAgent):
    agent_type = "statistician"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("statistician.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=STAT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response for now
        output = {
            "key_findings": ["Strong positive correlation between feature A and B"],
            "correlations": {"A_B": 0.85},
            "statistical_tests": [{"test": "t-test", "p_value": 0.01}],
        }

        self._log.info("statistician.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Performed correlation analysis and found significance.",
        )
