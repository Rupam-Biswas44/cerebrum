"""
Base Agent — Abstract class for all Cerebrum agents.

Every agent in the system inherits from BaseAgent.
This enforces a consistent interface for:
  - Input/output contracts (AgentContext → AgentResult)
  - Execution lifecycle (plan → retrieve → reason → execute → evaluate)
  - Observability (structured logging, metrics, tracing)
  - Retry and fallback logic
  - Token and cost tracking
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Possible states of an agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class LLMConfig:
    """Configuration for the LLM used by an agent."""

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 60
    max_retries: int = 3


@dataclass
class MemorySnapshot:
    """Snapshot of agent memory at the time of execution."""

    short_term: list[dict[str, Any]] = field(default_factory=list)
    relevant_long_term: list[dict[str, Any]] = field(default_factory=list)
    knowledge_nodes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SubTask:
    """A single subtask decomposed from a larger goal."""

    id: str
    description: str
    agent_type: str
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for how this agent's output should be evaluated."""

    check_faithfulness: bool = True
    check_hallucination: bool = True
    hallucination_threshold: float = 0.7
    require_evidence: bool = True
    min_confidence: float = 0.5


@dataclass
class EvaluationScore:
    """
    Evaluation scores for a single agent execution.
    All scores are in range [0.0, 1.0] unless noted.
    """

    task_completion: float = 0.0  # Did agent complete its task?
    faithfulness: float = 0.0  # Is output grounded in evidence?
    hallucination_rate: float = 0.0  # % of claims without evidence (lower is better)
    confidence: float = 0.0  # Agent's self-assessed confidence
    user_rating: float | None = None  # Optional human feedback

    @property
    def overall_score(self) -> float:
        """Weighted composite score."""
        return (
            self.task_completion * 0.4
            + self.faithfulness * 0.3
            + (1.0 - self.hallucination_rate) * 0.2
            + self.confidence * 0.1
        )


@dataclass
class AgentContext:
    """
    The input context passed to every agent.
    Contains everything the agent needs to execute its task.
    """

    task_id: str
    user_id: str
    project_id: str
    goal: str  # High-level user goal
    subtask: SubTask  # Specific task for this agent
    memory: MemorySnapshot  # Relevant memory
    tools: list[str]  # Available tool names
    llm_config: LLMConfig  # LLM configuration
    previous_outputs: dict[str, Any]  # Outputs from upstream agents
    evaluation_config: EvaluationConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AgentResult:
    """
    The standardized output of every agent execution.
    Includes output, reasoning chain, evidence, and evaluation scores.
    """

    agent_id: str
    agent_type: str
    task_id: str
    subtask_id: str
    status: AgentStatus

    # Core output
    output: Any = None  # The actual result
    reasoning: str = ""  # Chain-of-thought explanation
    evidence: list[str] = field(default_factory=list)  # Supporting evidence

    # Performance metrics
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0

    # Quality
    evaluation: EvaluationScore = field(default_factory=EvaluationScore)

    # Error information
    error: str | None = None
    error_type: str | None = None
    retries: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all Cerebrum agents.

    Subclasses must implement:
      - agent_type: class attribute identifying the agent
      - execute_task(): the core task execution logic

    The run() method provides the execution lifecycle:
      1. Validate context
      2. Log start
      3. Retrieve relevant memory
      4. Execute task (with retry)
      5. Evaluate output
      6. Log result
      7. Return AgentResult
    """

    agent_type: str = "base"

    def __init__(self, agent_id: str | None = None) -> None:
        self.agent_id = agent_id or f"{self.agent_type}-{uuid.uuid4().hex[:8]}"
        self._log = structlog.get_logger(self.__class__.__name__).bind(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
        )

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Main execution entry point. Handles the full lifecycle.
        Do NOT override this method — override execute_task() instead.
        """
        start_time = time.perf_counter()

        self._log.info(
            "agent.run.start",
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            goal=context.goal[:100],
        )

        with tracer.start_as_current_span(
            f"agent.{self.agent_type}.run",
            attributes={
                "agent.id": self.agent_id,
                "agent.type": self.agent_type,
                "task.id": context.task_id,
            },
        ):
            result = AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task_id=context.task_id,
                subtask_id=context.subtask.id,
                status=AgentStatus.RUNNING,
            )

            try:
                # Step 1: Validate input
                self._validate_context(context)

                # Step 2: Execute core task with retry
                result = await self._execute_with_retry(context, result)

                # Step 3: Evaluate output quality
                result.evaluation = await self._evaluate_output(context, result)

                # Step 4: Mark complete
                result.status = AgentStatus.COMPLETED

            except TimeoutError:
                result.status = AgentStatus.TIMEOUT
                result.error = f"Agent timed out after {context.llm_config.timeout_seconds}s"
                result.error_type = "TimeoutError"
                self._log.error("agent.run.timeout", task_id=context.task_id)

            except Exception as e:
                result.status = AgentStatus.FAILED
                result.error = str(e)
                result.error_type = type(e).__name__
                self._log.error(
                    "agent.run.failed",
                    task_id=context.task_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            finally:
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                self._log.info(
                    "agent.run.complete",
                    task_id=context.task_id,
                    status=result.status,
                    latency_ms=f"{result.latency_ms:.1f}",
                    tokens_used=result.tokens_used,
                    overall_score=f"{result.evaluation.overall_score:.3f}",
                )

        return result

    @abstractmethod
    async def execute_task(self, context: AgentContext) -> AgentResult:
        """
        Core task execution logic.
        Must be implemented by every agent subclass.

        Args:
            context: The agent execution context.

        Returns:
            AgentResult with output, reasoning, and evidence populated.
        """
        ...

    async def _execute_with_retry(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Execute with exponential backoff retry on transient errors."""
        max_retries = context.llm_config.max_retries
        delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                return await self.execute_task(context)
            except (TimeoutError, KeyboardInterrupt):
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                result.retries += 1
                self._log.warning(
                    "agent.retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                import asyncio

                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        return result  # Never reached, but satisfies type checker

    async def _evaluate_output(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> EvaluationScore:
        """
        Evaluate the quality of the agent's output.
        Subclasses can override for domain-specific evaluation.
        """
        score = EvaluationScore()

        # Task completion: was there output and no error?
        score.task_completion = 1.0 if (result.output is not None and result.error is None) else 0.0

        # Faithfulness: does output cite evidence?
        if context.evaluation_config.check_faithfulness:
            score.faithfulness = min(1.0, len(result.evidence) / 3) if result.evidence else 0.3

        # Hallucination: placeholder — overridden by CriticAgent for real analysis
        score.hallucination_rate = 0.0

        # Confidence: derived from faithfulness and task completion
        score.confidence = (score.task_completion + score.faithfulness) / 2

        return score

    def _validate_context(self, context: AgentContext) -> None:
        """Validate the agent context before execution."""
        if not context.goal:
            msg = "AgentContext.goal must not be empty"
            raise ValueError(msg)
        if not context.subtask:
            msg = "AgentContext.subtask must not be None"
            raise ValueError(msg)
        if not context.task_id:
            msg = "AgentContext.task_id must not be empty"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id!r})"
