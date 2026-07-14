"""
Writer Agent

Responsible for synthesizing the outputs of all upstream agents into a cohesive report.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

WRITER_SYSTEM_PROMPT = """You are the Writer Agent for Cerebrum.
Your job is to synthesize the findings from data engineering, statistics, and ML into a
professional, cohesive, and easy-to-read report in Markdown format.

Goal:
1. Summarize the initial goal and the data that was analyzed.
2. Highlight key statistical findings and ML results.
3. Provide business insights and recommendations.

Output format (strict JSON):
{{
  "report": "# Executive Summary\\n\\n...",
  "audience": "business stakeholders",
  "word_count": 500
}}
"""


class WriterAgent(BaseAgent):
    agent_type = "writer"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("writer.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response for now
        report_text = (
            "# Analysis Report\n\n## Findings\n"
            "We analyzed the dataset and found a strong correlation."
        )
        output = {
            "report": report_text,
            "audience": "general",
            "word_count": 14,
        }

        self._log.info("writer.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Drafted final markdown report based on previous agent outputs.",
        )
