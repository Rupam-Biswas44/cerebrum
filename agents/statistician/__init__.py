"""
Statistician Agent

Responsible for Exploratory Data Analysis (EDA), hypothesis testing,
correlation analysis, outlier detection, and NL-to-SQL execution.
Delegates to AnalyticsService and core.llm.sql_generator.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from core.llm.sql_generator import generate_sql_from_nl
from services.analytics import AnalyticsService

logger = structlog.get_logger(__name__)

STAT_SYSTEM_PROMPT = """You are the Statistician Agent for Cerebrum.
Given the user's goal and a summary of the data insights (correlations, outliers, sql results),
produce a final set of key findings.

Output format (strict JSON):
{{
  "key_findings": ["finding 1", "finding 2"],
  "summary": "Overall the dataset shows strong relationships in..."
}}
"""


class StatisticianAgent(BaseAgent):
    agent_type = "statistician"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("statistician.execute.start", subtask=context.subtask.id)

        analytics_service = AnalyticsService()

        # Step 1: Load data
        # In production this would load the real dataset from MinIO/DuckDB.
        dataset = context.previous_outputs.get("training_dataset")
        schema = context.previous_outputs.get("schema_summary", "{}")

        if dataset and isinstance(dataset, dict):
            df = pd.DataFrame.from_dict(dataset)
        else:
            # Generate dummy data for testing if no previous dataset
            import numpy as np

            df = pd.DataFrame(
                {
                    "price": np.random.normal(100, 20, 100),
                    "demand": np.random.normal(50, 10, 100),
                    "category": np.random.choice(["A", "B", "C"], 100),
                }
            )
            # Add some correlation
            df["revenue"] = df["price"] * df["demand"] * 0.5
            schema = {"price": "float", "demand": "float", "category": "string", "revenue": "float"}

        # Step 2: Automated Insights (Outliers & Correlations)
        insights: dict[str, Any] = {}
        try:
            insights = analytics_service.generate_insights(df)
            self._log.info("statistician.insights.generated")
        except Exception as e:
            self._log.warning("statistician.insights.failed", error=str(e))
            insights = {"error": str(e)}

        # Step 3: NL-to-SQL if subtask implies querying specific data
        sql_results = None
        sql_query = None
        sql_reasoning = None

        # We can trigger SQL if the user's goal contains keywords,
        # or we can just always try to answer it via SQL.
        # Let's try to answer the specific subtask description using SQL.
        try:
            sql_response = await generate_sql_from_nl(
                question=f"Goal: {context.goal} | Subtask: {context.subtask.description}",
                schema=schema,
                llm_model=context.llm_config.model,
                temperature=0.0,
            )
            sql_query = sql_response.query
            sql_reasoning = sql_response.reasoning

            # Execute the generated SQL
            if sql_query:
                sql_results = analytics_service.execute_sql(df, sql_query)
                self._log.info("statistician.sql.executed")

        except Exception as e:
            self._log.warning("statistician.sql.failed", error=str(e))
            sql_results = [{"error": str(e)}]

        # Step 4: Final LLM synthesis
        llm = ChatOpenAI(
            model=context.llm_config.model,
            temperature=context.llm_config.temperature,
            max_tokens=context.llm_config.max_tokens,
        )

        user_message = (
            f"Goal: {context.goal}\n"
            f"Subtask: {context.subtask.description}\n\n"
            f"Automated Insights (Correlations & Outliers):\n{json.dumps(insights)[:1000]}...\n\n"
            f"SQL Query Executed: {sql_query}\n"
            f"SQL Results:\n{json.dumps(sql_results)[:1000]}..."
        )

        messages = [
            SystemMessage(content=STAT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response: AIMessage = await llm.ainvoke(messages)
            llm_output = json.loads(str(response.content).strip())
        except Exception as e:
            self._log.warning("statistician.llm.failed", error=str(e))
            llm_output = {
                "key_findings": ["Failed to parse LLM response."],
                "summary": "Error during analysis synthesis.",
            }

        output = {
            "insights": insights,
            "sql_executed": sql_query,
            "sql_reasoning": sql_reasoning,
            "sql_results": sql_results,
            "synthesis": llm_output,
        }

        self._log.info("statistician.execute.complete")

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning=llm_output.get("summary", "Analysis complete."),
        )
