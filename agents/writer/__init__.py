"""
Writer Agent

Responsible for synthesizing the outputs of all upstream agents into a cohesive report.
Delegates to ReportingService to generate Markdown, PDF, and PPTX artifacts.
"""

from __future__ import annotations

import json

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from services.reporting import ReportingService

logger = structlog.get_logger(__name__)

WRITER_SYSTEM_PROMPT = """You are the Writer Agent for Cerebrum.
Your job is to synthesize the findings from data engineering, statistics, ML, and visualizations
into a professional, cohesive executive summary.

Goal:
1. Provide a narrative executive summary of the findings.
2. Extract the most important "key_findings" as bullet points.

Output format (strict JSON):
{
  "title": "A short, professional title for the report",
  "narrative": "Detailed executive summary...",
  "key_findings": ["finding 1", "finding 2"]
}
"""


class WriterAgent(BaseAgent):
    agent_type = "writer"

    async def execute_task(self, context: AgentContext) -> AgentResult:
        self._log.info("writer.execute.start", subtask=context.subtask.id)

        # ── Step 1: Generate Content via LLM ──────────────────────────────────
        llm = ChatOpenAI(
            model=context.llm_config.model,
            temperature=0.2,  # slightly more creative
            max_tokens=2048,
        )

        user_message = f"Goal: {context.goal}\nSubtask: {context.subtask.description}"
        if context.previous_outputs:
            # We truncate large arrays to prevent context overflow
            safe_outputs = str(context.previous_outputs)[:20000]
            user_message += f"\n\nPrevious outputs (truncated):\n{safe_outputs}"

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        try:
            response: AIMessage = await llm.ainvoke(messages)
            llm_output = json.loads(str(response.content).strip())
            title = llm_output.get("title", "Analysis Report")
            narrative = llm_output.get("narrative", "Analysis complete.")
            key_findings = llm_output.get("key_findings", ["Analysis completed successfully."])
        except Exception as e:
            self._log.warning("writer.llm.failed", error=str(e))
            title = "Analysis Report"
            narrative = "Failed to generate report narrative."
            key_findings = ["Error during synthesis."]

        # ── Step 2: Generate Report Artifacts ─────────────────────────────────
        reporting_service = ReportingService()
        generated_artifacts = {}

        # 1. Markdown
        try:
            md_content = reporting_service.generate_markdown(title, narrative, key_findings)
            md_url = reporting_service.upload_report_to_minio(
                md_content, f"{context.project_id}/{context.task_id}/report.md", "text/markdown"
            )
            if md_url:
                generated_artifacts["markdown"] = md_url
        except Exception as e:
            self._log.error("writer.markdown.failed", error=str(e))

        # 2. PDF
        try:
            pdf_bytes = reporting_service.export_to_pdf(title, narrative, key_findings)
            if pdf_bytes:
                pdf_url = reporting_service.upload_report_to_minio(
                    pdf_bytes,
                    f"{context.project_id}/{context.task_id}/report.pdf",
                    "application/pdf",
                )
                if pdf_url:
                    generated_artifacts["pdf"] = pdf_url
        except Exception as e:
            self._log.error("writer.pdf.failed", error=str(e))

        # 3. PPTX
        try:
            pptx_bytes = reporting_service.export_to_pptx(title, narrative, key_findings)
            if pptx_bytes:
                pptx_url = reporting_service.upload_report_to_minio(
                    pptx_bytes,
                    f"{context.project_id}/{context.task_id}/presentation.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
                if pptx_url:
                    generated_artifacts["pptx"] = pptx_url
        except Exception as e:
            self._log.error("writer.pptx.failed", error=str(e))

        output = {
            "title": title,
            "narrative": narrative,
            "key_findings": key_findings,
            "artifacts": generated_artifacts,
        }

        self._log.info("writer.execute.complete", artifacts_generated=len(generated_artifacts))

        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=context.task_id,
            subtask_id=context.subtask.id,
            status=AgentStatus.COMPLETED,
            output=output,
            reasoning=f"Report generated with {len(generated_artifacts)} artifacts.",
        )
