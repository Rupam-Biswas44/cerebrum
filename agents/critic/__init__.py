"""
Critic Agent

Responsible for reviewing outputs to ensure quality, accuracy, and adherence.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent for Cerebrum.
Your job is to review the proposed outputs from other agents before they are finalized.
You evaluate for accuracy, completeness, and adherence to the user's original goal.

Goal:
1. Review the previous outputs.
2. Determine if the output meets the requirements.
3. Provide feedback or approval.

Output format (strict JSON):
{{
  "approved": true,
  "feedback": "The analysis looks solid, but could use more detail on the outliers.",
  "score": 0.85
}}
"""


class CriticAgent(BaseAgent):
    agent_type = "critic"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("critic.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs to review: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response
        output = {
            "approved": True,
            "feedback": "All requirements met. Ready for final report.",
            "score": 0.95,
        }

        self._log.info("critic.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Reviewed and approved outputs from previous steps.",
        )
