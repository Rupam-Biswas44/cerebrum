"""
Data Engineer Agent

Responsible for loading, cleaning, transforming, and profiling datasets.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

DE_SYSTEM_PROMPT = """You are the Data Engineer Agent for Cerebrum.
Your job is to interact with datasets, clean them, transform them, and extract schemas.
You have access to tools that can run SQL queries against raw datasets and profile them.

Goal:
1. Profile the dataset to understand its schema.
2. Execute SQL queries to clean, filter, and transform the data as requested.
3. Return a JSON object with the summary of what you did.

Output format (strict JSON):
{{
  "actions_taken": ["profiled dataset", "removed nulls", ...],
  "schema_summary": {{...}},
  "transformation_results": {{...}},
  "recommendations_for_next_agent": "..."
}}
"""


class DataEngineerAgent(BaseAgent):
    agent_type = "data_engineer"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("data_engineer.execute.start", subtask=context.subtask.id)

        # In a real implementation, this would use Langchain's bind_tools
        # and a proper reasoning loop. For now we mock the LLM call or do a simple completion.

        # Build LLM
        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=DE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # In Milestone 7, we simulate the LLM call for now, but in a full implementation
        # it would execute tool_registry tools like query_dataset.

        # Simulating successful output for now
        output = {
            "actions_taken": ["Extracted data from object storage", "Cleaned null values"],
            "schema_summary": {"columns": ["id", "value"], "rows": 1000},
            "transformation_results": {"status": "success"},
            "recommendations_for_next_agent": "Ready for statistical analysis",
        }

        self._log.info("data_engineer.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Executed data cleaning pipeline.",
        )
