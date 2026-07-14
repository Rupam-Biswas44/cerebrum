"""
LangGraph Orchestration Engine

The core multi-agent workflow engine built on LangGraph.

Architecture:
  LangGraph treats a multi-agent pipeline as a directed graph where:
    - Each NODE is an agent (Planner, DataEngineer, Statistician, etc.)
    - Each EDGE represents when control passes to the next agent
    - Conditional edges enable dynamic routing based on task type

  Workflow for a typical analysis task:
    ┌──────────┐     ┌─────────────┐     ┌──────────────┐
    │ Planner  │────▶│DataEngineer │────▶│ Statistician │
    └──────────┘     └─────────────┘     └──────────────┘
         │                                       │
         │           ┌──────────────┐            │
         └──────────▶│  MLEngineer  │            │
                     └──────────────┘            │
                            │                    ▼
                     ┌──────────────┐    ┌──────────────┐
                     │Visualization │◀───│   Critic     │
                     └──────────────┘    └──────────────┘
                            │
                     ┌──────────────┐
                     │    Writer    │
                     └──────────────┘
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

logger = structlog.get_logger(__name__)


# ============================================================
# Shared Graph State
# ============================================================


class OrchestratorState(TypedDict):
    """
    Shared mutable state that flows through every node in the graph.
    Each agent reads from and writes to this state.
    """

    # Task identity
    task_id: str
    project_id: str
    user_id: str
    goal: str

    # Plan produced by the PlannerAgent
    plan: dict[str, Any]
    subtasks: list[dict[str, Any]]

    # Accumulated outputs from each agent
    agent_outputs: dict[str, Any]

    # Routing control
    current_agent: str
    next_agents: list[str]
    task_type: str  # 'analysis', 'ml_training', 'report', 'custom'

    # Final result
    final_report: str
    error: str | None
    completed: bool


# ============================================================
# Node Implementations (Graph Nodes)
# ============================================================


async def planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Planner Node — decomposes the user's goal into a structured plan.
    Determines which agents are needed and in what order.
    """
    logger.info("orchestrator.node.planner", task_id=state["task_id"])

    try:
        from agents.base import AgentContext, LLMConfig
        from agents.planner import PlannerAgent

        agent = PlannerAgent(llm_config=LLMConfig())
        context = AgentContext(
            task_id=uuid.UUID(state["task_id"]),
            project_id=uuid.UUID(state["project_id"]),
            user_id=uuid.UUID(state["user_id"]),
            goal=state["goal"],
        )
        result = await agent.run(context)

        plan = result.output.get("plan", {})
        subtasks = result.output.get("subtasks", [])

        # Determine which specialist agents are needed
        needed_agents = _determine_agent_pipeline(plan, state["task_type"])

        return {
            **state,
            "plan": plan,
            "subtasks": subtasks,
            "next_agents": needed_agents,
            "agent_outputs": {"planner": result.output},
            "current_agent": "planner",
        }
    except Exception as e:
        logger.error("orchestrator.planner.failed", error=str(e))
        return {**state, "error": f"Planning failed: {e}", "completed": True}


async def data_engineer_node(state: OrchestratorState) -> OrchestratorState:
    """Data Engineer Node — queries, cleans, and transforms datasets."""
    logger.info("orchestrator.node.data_engineer", task_id=state["task_id"])
    try:
        from agents.base import AgentContext, LLMConfig
        from agents.data_engineer import DataEngineerAgent

        agent = DataEngineerAgent(llm_config=LLMConfig())
        context = AgentContext(
            task_id=uuid.UUID(state["task_id"]),
            project_id=uuid.UUID(state["project_id"]),
            user_id=uuid.UUID(state["user_id"]),
            goal=state["goal"],
            plan=state["plan"],
            previous_outputs=state["agent_outputs"],
        )
        result = await agent.run(context)
        return {
            **state,
            "agent_outputs": {**state["agent_outputs"], "data_engineer": result.output},
            "current_agent": "data_engineer",
        }
    except Exception as e:
        logger.error("orchestrator.data_engineer.failed", error=str(e))
        return {**state, "error": f"Data engineering failed: {e}", "completed": True}


async def statistician_node(state: OrchestratorState) -> OrchestratorState:
    """Statistician Node — performs EDA, hypothesis testing, and correlations."""
    logger.info("orchestrator.node.statistician", task_id=state["task_id"])
    try:
        from agents.base import AgentContext, LLMConfig
        from agents.statistician import StatisticianAgent

        agent = StatisticianAgent(llm_config=LLMConfig())
        context = AgentContext(
            task_id=uuid.UUID(state["task_id"]),
            project_id=uuid.UUID(state["project_id"]),
            user_id=uuid.UUID(state["user_id"]),
            goal=state["goal"],
            plan=state["plan"],
            previous_outputs=state["agent_outputs"],
        )
        result = await agent.run(context)
        return {
            **state,
            "agent_outputs": {**state["agent_outputs"], "statistician": result.output},
            "current_agent": "statistician",
        }
    except Exception as e:
        logger.error("orchestrator.statistician.failed", error=str(e))
        return {**state, "error": f"Statistics failed: {e}", "completed": True}


async def writer_node(state: OrchestratorState) -> OrchestratorState:
    """Writer Node — synthesizes all outputs into a final report."""
    logger.info("orchestrator.node.writer", task_id=state["task_id"])
    try:
        from agents.base import AgentContext, LLMConfig
        from agents.writer import WriterAgent

        agent = WriterAgent(llm_config=LLMConfig())
        context = AgentContext(
            task_id=uuid.UUID(state["task_id"]),
            project_id=uuid.UUID(state["project_id"]),
            user_id=uuid.UUID(state["user_id"]),
            goal=state["goal"],
            plan=state["plan"],
            previous_outputs=state["agent_outputs"],
        )
        result = await agent.run(context)
        return {
            **state,
            "agent_outputs": {**state["agent_outputs"], "writer": result.output},
            "final_report": result.output.get("report", ""),
            "current_agent": "writer",
            "completed": True,
        }
    except Exception as e:
        logger.error("orchestrator.writer.failed", error=str(e))
        return {**state, "error": f"Report writing failed: {e}", "completed": True}


# ============================================================
# Conditional Routing
# ============================================================


def _determine_agent_pipeline(plan: dict[str, Any], task_type: str) -> list[str]:
    """
    Based on the plan and task type, decide which agents to run.
    Returns an ordered list of agent names.
    """
    pipelines = {
        "analysis": ["data_engineer", "statistician", "writer"],
        "ml_training": ["data_engineer", "statistician", "ml_engineer", "writer"],
        "report": ["data_engineer", "writer"],
        "custom": ["data_engineer", "statistician", "writer"],
    }
    return pipelines.get(task_type, pipelines["analysis"])


def route_after_planner(state: OrchestratorState) -> str:
    """Conditional edge: after planning, decide the first specialist agent."""
    if state.get("error"):
        return END
    next_agents = state.get("next_agents", [])
    if not next_agents:
        return "writer"
    return next_agents[0]


def route_after_data_engineer(state: OrchestratorState) -> str:
    """Conditional edge: after data engineering, decide next step."""
    if state.get("error"):
        return END
    next_agents = state.get("next_agents", [])
    if "statistician" in next_agents:
        return "statistician"
    return "writer"


def route_after_statistician(state: OrchestratorState) -> str:
    """Conditional edge: after statistics, proceed to writer."""
    if state.get("error"):
        return END
    return "writer"


# ============================================================
# Graph Builder
# ============================================================


def build_orchestrator_graph() -> Any:
    """
    Build and compile the LangGraph orchestration graph.

    Returns a compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(OrchestratorState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("data_engineer", data_engineer_node)
    workflow.add_node("statistician", statistician_node)
    workflow.add_node("writer", writer_node)

    # Edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "data_engineer": "data_engineer",
            "statistician": "statistician",
            "writer": "writer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "data_engineer",
        route_after_data_engineer,
        {
            "statistician": "statistician",
            "writer": "writer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "statistician",
        route_after_statistician,
        {
            "writer": "writer",
            END: END,
        },
    )
    workflow.add_edge("writer", END)

    return workflow.compile()


async def run_orchestrator(
    task_id: str,
    project_id: str,
    user_id: str,
    goal: str,
    task_type: str = "analysis",
) -> OrchestratorState:
    """
    Entry point to kick off a multi-agent orchestration run.

    Args:
        task_id: Unique task ID (used for message bus and logging).
        project_id: The project namespace.
        user_id: The user who initiated the task.
        goal: The natural language goal/query.
        task_type: Pipeline type ('analysis', 'ml_training', 'report').

    Returns:
        The final OrchestratorState after all agents complete.
    """
    graph = build_orchestrator_graph()

    initial_state: OrchestratorState = {
        "task_id": task_id,
        "project_id": project_id,
        "user_id": user_id,
        "goal": goal,
        "plan": {},
        "subtasks": [],
        "agent_outputs": {},
        "current_agent": "",
        "next_agents": [],
        "task_type": task_type,
        "final_report": "",
        "error": None,
        "completed": False,
    }

    logger.info(
        "orchestrator.run.start",
        task_id=task_id,
        goal=goal[:100],
        task_type=task_type,
    )

    final_state: OrchestratorState = await graph.ainvoke(initial_state)  # type: ignore[assignment]

    logger.info(
        "orchestrator.run.complete",
        task_id=task_id,
        completed=final_state.get("completed"),
        error=final_state.get("error"),
    )

    return final_state
