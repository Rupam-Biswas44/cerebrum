"""
Unit Tests — BaseAgent

Tests the BaseAgent abstract class contract, lifecycle,
retry logic, and evaluation scoring without any external dependencies.
"""

from __future__ import annotations

import uuid

import pytest

from agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
    EvaluationConfig,
    EvaluationScore,
    LLMConfig,
    MemorySnapshot,
    SubTask,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_context() -> AgentContext:
    """A minimal valid AgentContext for testing."""
    return AgentContext(
        task_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        goal="Analyze sales data and find revenue trends",
        subtask=SubTask(
            id=str(uuid.uuid4()),
            description="Run statistical analysis on sales data",
            agent_type="statistician",
        ),
        memory=MemorySnapshot(),
        tools=["pandas", "statistical_analysis"],
        llm_config=LLMConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.1,
            max_retries=2,
        ),
        previous_outputs={},
        evaluation_config=EvaluationConfig(),
    )


@pytest.fixture
def successful_agent() -> BaseAgent:
    """A concrete agent that always succeeds."""

    class SuccessAgent(BaseAgent):
        agent_type = "test_success"

        async def execute_task(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task_id=context.task_id,
                subtask_id=context.subtask.id,
                status=AgentStatus.COMPLETED,
                output={"result": "analysis complete", "rows_analyzed": 1000},
                reasoning="Analyzed the data using statistical methods",
                evidence=["Data contains 1000 rows", "Revenue trend is negative Q3"],
                tokens_used=500,
            )

    return SuccessAgent()


@pytest.fixture
def failing_agent() -> BaseAgent:
    """A concrete agent that always fails."""

    class FailAgent(BaseAgent):
        agent_type = "test_fail"
        call_count = 0

        async def execute_task(self, context: AgentContext) -> AgentResult:
            self.call_count += 1
            msg = "LLM API error: rate limit exceeded"
            raise ConnectionError(msg)

    return FailAgent()


@pytest.fixture
def flaky_agent() -> BaseAgent:
    """A concrete agent that fails twice then succeeds."""

    class FlakyAgent(BaseAgent):
        agent_type = "test_flaky"
        call_count = 0

        async def execute_task(self, context: AgentContext) -> AgentResult:
            self.call_count += 1
            if self.call_count < 3:
                msg = f"Transient error on attempt {self.call_count}"
                raise ConnectionError(msg)
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task_id=context.task_id,
                subtask_id=context.subtask.id,
                status=AgentStatus.COMPLETED,
                output={"result": "recovered after retries"},
            )

    return FlakyAgent()


# ── Tests: Agent Lifecycle ────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.agent
class TestAgentLifecycle:
    """Tests for the BaseAgent run() lifecycle."""

    @pytest.mark.asyncio
    async def test_successful_agent_returns_completed_status(
        self,
        successful_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """A successful agent should return COMPLETED status."""
        result = await successful_agent.run(sample_context)

        assert result.status == AgentStatus.COMPLETED
        assert result.error is None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_successful_agent_populates_metrics(
        self,
        successful_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Agent should populate latency and tokens."""
        result = await successful_agent.run(sample_context)

        assert result.latency_ms > 0
        assert result.tokens_used == 500
        assert result.agent_id == successful_agent.agent_id
        assert result.agent_type == "test_success"

    @pytest.mark.asyncio
    async def test_successful_agent_evaluates_output(
        self,
        successful_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Agent run should trigger output evaluation."""
        result = await successful_agent.run(sample_context)

        assert result.evaluation.task_completion == 1.0
        assert result.evaluation.faithfulness > 0  # Has evidence
        assert 0.0 <= result.evaluation.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_failed_agent_returns_failed_status(
        self,
        failing_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """A failing agent should return FAILED status, not raise."""
        # LLMConfig.max_retries=2 means 3 total attempts
        result = await failing_agent.run(sample_context)

        assert result.status == AgentStatus.FAILED
        assert result.error is not None
        assert "rate limit" in result.error.lower()
        assert result.error_type == "ConnectionError"

    @pytest.mark.asyncio
    async def test_failed_agent_records_retries(
        self,
        failing_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Failed agent should record retry count."""
        result = await failing_agent.run(sample_context)
        assert result.retries == 2  # max_retries=2

    @pytest.mark.asyncio
    async def test_flaky_agent_recovers_with_retry(
        self,
        flaky_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Agent should recover from transient errors via retry."""
        result = await flaky_agent.run(sample_context)

        assert result.status == AgentStatus.COMPLETED
        assert result.retries == 2  # Failed twice, succeeded on 3rd attempt
        assert result.output["result"] == "recovered after retries"

    @pytest.mark.asyncio
    async def test_agent_always_returns_latency(
        self,
        failing_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Even failed agents must have latency populated."""
        result = await failing_agent.run(sample_context)
        assert result.latency_ms > 0


# ── Tests: Context Validation ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.agent
class TestContextValidation:
    """Tests for AgentContext validation."""

    @pytest.mark.asyncio
    async def test_empty_goal_causes_failure(
        self,
        successful_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Empty goal should cause FAILED status."""
        sample_context.goal = ""
        result = await successful_agent.run(sample_context)

        assert result.status == AgentStatus.FAILED
        assert "goal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_task_id_causes_failure(
        self,
        successful_agent: BaseAgent,
        sample_context: AgentContext,
    ) -> None:
        """Empty task_id should cause FAILED status."""
        sample_context.task_id = ""
        result = await successful_agent.run(sample_context)

        assert result.status == AgentStatus.FAILED


# ── Tests: Agent Identity ─────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.agent
class TestAgentIdentity:
    """Tests for agent identity and naming."""

    def test_agent_auto_generates_id(self, successful_agent: BaseAgent) -> None:
        """Agent should auto-generate a unique ID if none provided."""
        assert successful_agent.agent_id.startswith("test_success-")
        assert len(successful_agent.agent_id) > len("test_success-")

    def test_agent_accepts_custom_id(self) -> None:
        """Agent should use provided ID."""

        class MyAgent(BaseAgent):
            agent_type = "custom"

            async def execute_task(self, context: AgentContext) -> AgentResult:
                return AgentResult(  # type: ignore
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task_id="",
                    subtask_id="",
                    status=AgentStatus.COMPLETED,
                )

        agent = MyAgent(agent_id="my-custom-id-123")
        assert agent.agent_id == "my-custom-id-123"

    def test_agent_repr(self, successful_agent: BaseAgent) -> None:
        """Agent repr should include class name and ID."""
        repr_str = repr(successful_agent)
        assert "SuccessAgent" in repr_str
        assert successful_agent.agent_id in repr_str


# ── Tests: Evaluation Scoring ─────────────────────────────────────────────


@pytest.mark.unit
class TestEvaluationScore:
    """Tests for EvaluationScore calculation."""

    def test_perfect_score(self) -> None:
        score = EvaluationScore(
            task_completion=1.0,
            faithfulness=1.0,
            hallucination_rate=0.0,
            confidence=1.0,
        )
        assert score.overall_score == pytest.approx(1.0, abs=0.001)

    def test_zero_score(self) -> None:
        score = EvaluationScore(
            task_completion=0.0,
            faithfulness=0.0,
            hallucination_rate=1.0,
            confidence=0.0,
        )
        assert score.overall_score == pytest.approx(0.0, abs=0.001)

    def test_partial_score(self) -> None:
        score = EvaluationScore(
            task_completion=1.0,
            faithfulness=0.8,
            hallucination_rate=0.1,
            confidence=0.9,
        )
        # task_completion(0.4) + faithfulness(0.24) + (1-hall)(0.18) + confidence(0.09) = 0.91
        assert 0.8 <= score.overall_score <= 1.0

    def test_score_bounds(self) -> None:
        """Overall score must always be between 0 and 1."""
        for _ in range(10):
            import secrets

            score = EvaluationScore(
                task_completion=secrets.SystemRandom().random(),
                faithfulness=secrets.SystemRandom().random(),
                hallucination_rate=secrets.SystemRandom().random(),
                confidence=secrets.SystemRandom().random(),
            )
            assert 0.0 <= score.overall_score <= 1.0
