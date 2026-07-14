"""
Visualization Agent

Responsible for determining the best chart types to represent the data,
then delegating to VisualizationService to generate Plotly figures and export them.
"""

from __future__ import annotations

import json

import pandas as pd
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from services.visualization import ChartSpec, VisualizationService

logger = structlog.get_logger(__name__)

VIS_SYSTEM_PROMPT = """You are the Visualization Agent for Cerebrum.
Given the user's goal, the subtask description, and the data schema,
determine the best visualizations to generate.

Output STRICT JSON containing a list of chart specifications.
Valid chart types are: 'scatter', 'line', 'bar', 'histogram', 'box', 'heatmap'.

Example Output:
{
  "charts": [
    {
      "chart_type": "scatter",
      "x_col": "price",
      "y_col": "demand",
      "color_col": "category",
      "title": "Price vs Demand by Category",
      "description": "Shows the relationship between price and demand."
    }
  ]
}
"""


class VisualizationAgent(BaseAgent):
    agent_type = "visualization"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("visualization.execute.start", subtask=context.subtask.id)

        # ── Step 1: Request Chart Specs from LLM ──────────────────────────────
        llm = ChatOpenAI(
            model=context.llm_config.model,
            temperature=0.0,
            max_tokens=1024,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            schema = context.previous_outputs.get("schema_summary", {})
            user_message += f"\nData Schema:\n{json.dumps(schema)}"

        messages = [
            SystemMessage(content=VIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response: AIMessage = await llm.ainvoke(messages)
            llm_output = json.loads(str(response.content).strip())
            chart_specs_raw = llm_output.get("charts", [])
        except Exception as e:
            self._log.warning("visualization.llm.failed", error=str(e))
            chart_specs_raw = []

        # ── Step 2: Load Data ──────────────────────────────────────────────────
        dataset = context.previous_outputs.get("training_dataset")
        if dataset and isinstance(dataset, dict):
            df = pd.DataFrame.from_dict(dataset)
        else:
            # Dummy dataset if none available
            import numpy as np

            df = pd.DataFrame(
                {
                    "price": np.random.normal(100, 20, 100),
                    "demand": np.random.normal(50, 10, 100),
                    "category": np.random.choice(["A", "B", "C"], 100),
                }
            )

        # ── Step 3: Generate Visualizations via Service ────────────────────────
        vis_service = VisualizationService()
        generated_charts = []

        for i, spec_dict in enumerate(chart_specs_raw):
            try:
                spec = ChartSpec(**spec_dict)
                fig = vis_service.generate_plotly_figure(df, spec)

                if fig is not None:
                    # 1. Export as JSON for interactive frontend
                    fig_json = vis_service.export_figure_to_json(fig)

                    # 2. Upload to MinIO as PNG
                    object_name = f"{context.project_id}/{context.task_id}/chart_{i}.png"
                    image_url = vis_service.upload_figure_to_minio(fig, object_name)

                    generated_charts.append(
                        {
                            "spec": spec_dict,
                            "plotly_json": fig_json,
                            "image_url": image_url,
                        }
                    )
                    self._log.info("visualization.chart.generated", chart=spec.title)
            except Exception as e:
                self._log.error("visualization.chart.failed", spec=spec_dict, error=str(e))

        output = {"generated_charts": generated_charts, "total_charts": len(generated_charts)}

        self._log.info("visualization.execute.complete", charts=len(generated_charts))

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning=f"Generated {len(generated_charts)} charts based on data schema.",
        )
