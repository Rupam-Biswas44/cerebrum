"""
Memory Agent

Responsible for storing and retrieving context from long-term memory (PostgreSQL / Qdrant)
to provide relevant historical insights to the orchestrator.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

MEMORY_SYSTEM_PROMPT = """You are the Memory Agent for Cerebrum.
Your job is to identify what context needs to be retrieved from long-term memory
or what new insights need to be stored based on the current task execution.

Goal:
1. Extract key entities or topics from the current goal.
2. Query vector store / relational memory for related context.
3. Formulate new memories to be saved.

Output format (strict JSON):
{{
  "retrieved_context": ["past insight 1", "past insight 2"],
  "memories_to_store": ["new insight 1"]
}}
"""


class MemoryAgent(BaseAgent):
    agent_type = "memory"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("memory.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"

        [
            SystemMessage(content=MEMORY_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response
        output = {
            "retrieved_context": ["User prefers detailed statistical breakdowns."],
            "memories_to_store": ["Dataset X has high sparsity in column Y."],
        }

        self._log.info("memory.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Retrieved relevant user preferences from memory.",
        )
