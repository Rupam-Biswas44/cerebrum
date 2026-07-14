"""
Visualization Agent

Responsible for generating code (Plotly, Vega-Lite, Matplotlib) for data visualization.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent

logger = structlog.get_logger(__name__)

VIS_SYSTEM_PROMPT = """You are the Visualization Agent for Cerebrum.
Your job is to generate visualization specifications or code based on data and statistical findings.

Goal:
1. Determine the best chart types to represent the data.
2. Generate JSON configurations (e.g., Vega-Lite) or Python code (e.g., Plotly) for the charts.
3. Ensure the charts are aesthetically pleasing and professional.

Output format (strict JSON):
{{
  "charts": [
    {{
      "title": "Chart Title",
      "type": "bar_chart",
      "spec": {{...}}
    }}
  ]
}}
"""


class VisualizationAgent(BaseAgent):
    agent_type = "visualization"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("visualization.execute.start", subtask=context.subtask.id)

        ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            user_message += f"\nPrevious outputs: {json.dumps(context.previous_outputs)}"

        [
            SystemMessage(content=VIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Simulating LLM response for now
        output = {
            "charts": [
                {
                    "title": "Correlation Matrix",
                    "type": "heatmap",
                    "spec": {"data": [], "layout": {}},
                }
            ]
        }

        self._log.info("visualization.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning="Generated a heatmap for the correlation matrix.",
        )
