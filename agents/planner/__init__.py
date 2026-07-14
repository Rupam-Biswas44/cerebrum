"""
Planner Agent — Goal Decomposition

The Planner Agent is the entry point of every Cerebrum workflow.
It takes a high-level user goal and decomposes it into a structured
execution plan with ordered subtasks assigned to specialized agents.

Example:
    Input:  "Analyze Q3 sales, find revenue drop, predict Q4, make report"
    Output: [
        SubTask(agent="data_engineer", description="Load and validate sales CSV"),
        SubTask(agent="statistician", description="Run EDA and correlation analysis"),
        SubTask(agent="ml_engineer", description="Train Q4 sales forecast model"),
        SubTask(agent="visualization", description="Generate revenue trend charts"),
        SubTask(agent="writer", description="Write executive summary report"),
        SubTask(agent="critic", description="Review and fact-check the report"),
    ]
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
    EvaluationScore,
    SubTask,
)

logger = structlog.get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent for Cerebrum, an enterprise AI platform.
Your role is to decompose a user's high-level goal into an ordered list of subtasks,
each assigned to a specialized agent.

Available agents and their capabilities:
- data_engineer: Load, validate, clean, profile, and transform data files
- statistician: Statistical analysis, EDA, hypothesis testing, correlation analysis
- ml_engineer: Build ML models (classification, regression, forecasting, anomaly detection)
- visualization: Create interactive charts, dashboards, and data visualizations
- writer: Write reports, summaries, presentations, and documentation
- critic: Fact-check, detect hallucinations, validate conclusions with evidence
- memory: Retrieve relevant past context and user preferences
- research: Search external sources and retrieve relevant information

Rules:
1. Create the minimum number of subtasks needed (max {max_subtasks})
2. Order subtasks by dependency (earlier tasks produce inputs for later tasks)
3. Data always flows: data_engineer → statistician/ml_engineer → visualization → writer → critic
4. Always include a critic agent as the last step for any report or analysis
5. Return ONLY valid JSON, no explanations

Output format:
{{
  "plan_reasoning": "Brief explanation of why you chose this plan",
  "subtasks": [
    {{
      "id": "unique-id",
      "description": "Specific, actionable task description",
      "agent_type": "agent_name",
      "priority": 0,
      "depends_on": ["id-of-dependency"],
      "metadata": {{}}
    }}
  ]
}}
"""


class PlannerAgent(BaseAgent):
    """
    Decomposes high-level goals into structured multi-agent execution plans.

    The Planner is called first in every workflow. It produces a DAG of
    SubTask objects that the orchestrator then executes in dependency order.
    """

    agent_type = "planner"

    def __init__(
        self,
        agent_id: str | None = None,
        max_subtasks: int = 15,
    ) -> None:
        super().__init__(agent_id)
        self.max_subtasks = max_subtasks

    async def execute_task(self, context: AgentContext) -> AgentResult:
        """
        Generate an execution plan from the user's goal.

        Args:
            context: AgentContext with the user's goal.

        Returns:
            AgentResult with output containing the list of SubTasks.
        """
        self._log.info("planner.decompose.start", goal=context.goal[:200])

        # Build the LLM
        llm = self._build_llm(context)

        # Construct messages
        system_prompt = PLANNER_SYSTEM_PROMPT.format(max_subtasks=self.max_subtasks)
        user_message = self._build_user_message(context)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        # Invoke LLM
        response = await llm.ainvoke(messages)
        raw_content = response.content

        # Parse the plan
        plan = self._parse_plan(raw_content)

        subtasks = [
            SubTask(
                id=st.get("id", str(uuid.uuid4())),
                description=st["description"],
                agent_type=st["agent_type"],
                priority=st.get("priority", i),
                depends_on=st.get("depends_on", []),
                metadata=st.get("metadata", {}),
            )
            for i, st in enumerate(plan["subtasks"])
        ]

        self._log.info(
            "planner.decompose.complete",
            subtask_count=len(subtasks),
            agents_involved=[st.agent_type for st in subtasks],
        )

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output={
                "subtasks": subtasks,
                "plan_reasoning": plan.get("plan_reasoning", ""),
                "total_steps": len(subtasks),
            },
            reasoning=plan.get("plan_reasoning", ""),
            evidence=["Generated from user goal via LLM planning"],
            tokens_used=response.usage_metadata.get("total_tokens", 0)
            if hasattr(response, "usage_metadata")
            else 0,
            evaluation=EvaluationScore(
                task_completion=1.0,
                faithfulness=0.9,
                confidence=0.85,
            ),
        )

    def _build_llm(self, context: AgentContext) -> ChatOpenAI:
        """Build the LLM client from context configuration."""
        return ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
            timeout=context.llm_config.timeout_seconds,
        )

    def _build_user_message(self, context: AgentContext) -> str:
        """Build the user message with full context."""
        parts = [f"Goal: {context.goal}"]

        if context.previous_outputs:
            parts.append(
                f"\nContext from previous agents:\n{json.dumps(context.previous_outputs, indent=2)[:1000]}"  # noqa: E501
            )

        if context.memory.short_term:
            recent = context.memory.short_term[-5:]  # Last 5 interactions
            parts.append(f"\nRecent conversation:\n{json.dumps(recent, indent=2)[:500]}")

        if context.metadata.get("available_datasets"):
            datasets = context.metadata["available_datasets"]
            parts.append(f"\nAvailable datasets: {', '.join(datasets)}")

        return "\n".join(parts)

    def _parse_plan(self, raw_content: str) -> dict[str, Any]:
        """
        Parse LLM response into a structured plan.
        Handles cases where the LLM wraps JSON in markdown code blocks.
        """
        content = raw_content.strip()

        # Strip markdown code block if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json) and last line (```)
            content = "\n".join(lines[1:-1])

        try:
            plan = json.loads(content)
        except json.JSONDecodeError as e:
            self._log.error("planner.parse.failed", error=str(e), content=content[:200])
            # Fallback: create a minimal single-step plan
            plan = {
                "plan_reasoning": "Fallback plan due to JSON parse error",
                "subtasks": [
                    {
                        "id": str(uuid.uuid4()),
                        "description": "Analyze the user's request",
                        "agent_type": "statistician",
                        "priority": 0,
                        "depends_on": [],
                        "metadata": {},
                    }
                ],
            }

        # Validate required fields
        if "subtasks" not in plan:
            msg = "Plan missing 'subtasks' field"
            raise ValueError(msg)

        # Cap to max_subtasks
        plan["subtasks"] = plan["subtasks"][: self.max_subtasks]

        return plan
