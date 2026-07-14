"""
Visualization Service

Generates interactive charts (Plotly JSON specifications) and static images (PNGs)
from Pandas DataFrames. Handles uploading these artifacts to MinIO.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import structlog
from pydantic import BaseModel, Field

from core.storage.minio import ensure_bucket_exists, minio_client

logger = structlog.get_logger(__name__)


class ChartSpec(BaseModel):
    """Specification for generating a chart."""

    chart_type: str = Field(
        ..., description="Type of chart: 'scatter', 'line', 'bar', 'histogram', 'box', 'heatmap'"
    )
    x_col: str | None = Field(None, description="Column for X axis")
    y_col: str | None = Field(None, description="Column for Y axis")
    color_col: str | None = Field(None, description="Column for color grouping")
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Brief description of what the chart shows")


class VisualizationService:
    """Service for generating charts and storing them."""

    def __init__(self, bucket_name: str = "visualizations"):
        self.bucket_name = bucket_name
        # Ensure bucket exists in a real deployment, safe to call multiple times
        if minio_client is not None:
            try:
                ensure_bucket_exists(self.bucket_name)
            except Exception as e:
                logger.warning("visualization.bucket.failed", error=str(e))

    def generate_plotly_figure(self, df: pd.DataFrame, spec: ChartSpec) -> go.Figure | None:
        """
        Generate a Plotly Figure object based on the specification and DataFrame.
        """
        try:
            if spec.chart_type == "scatter":
                fig = px.scatter(
                    df, x=spec.x_col, y=spec.y_col, color=spec.color_col, title=spec.title
                )
            elif spec.chart_type == "line":
                fig = px.line(
                    df, x=spec.x_col, y=spec.y_col, color=spec.color_col, title=spec.title
                )
            elif spec.chart_type == "bar":
                fig = px.bar(df, x=spec.x_col, y=spec.y_col, color=spec.color_col, title=spec.title)
            elif spec.chart_type == "histogram":
                fig = px.histogram(df, x=spec.x_col, color=spec.color_col, title=spec.title)
            elif spec.chart_type == "box":
                fig = px.box(df, x=spec.x_col, y=spec.y_col, color=spec.color_col, title=spec.title)
            elif spec.chart_type == "heatmap":
                # For heatmap, assume df is already a correlation matrix or 2D grid
                # If it has specific columns, we might need to pivot, but let's assume
                # the numeric correlation matrix case is handled prior.
                numeric_df = df.select_dtypes(include="number")
                fig = px.imshow(numeric_df.corr(), title=spec.title)
            else:
                logger.error("visualization.unsupported_chart", chart_type=spec.chart_type)
                return None

            # Standardize layout styling
            fig.update_layout(template="plotly_white", margin={"l": 40, "r": 40, "t": 60, "b": 40})
            return fig

        except Exception as e:
            logger.error("visualization.generate.failed", error=str(e), spec=spec.dict())
            return None

    def export_figure_to_json(self, fig: go.Figure) -> dict[str, Any]:
        """
        Export a Plotly Figure to a JSON-serializable dictionary.
        This is perfect for sending to a React frontend (react-plotly.js).
        """
        try:
            fig_json = fig.to_json()
            return json.loads(fig_json)
        except Exception as e:
            logger.error("visualization.export_json.failed", error=str(e))
            return {}

    def upload_figure_to_minio(
        self, fig: go.Figure, object_name: str, format: str = "png"
    ) -> str | None:
        """
        Export figure to a static image (PNG) and upload to MinIO.
        Returns the artifact URL or None.
        Requires kaleido to be installed.
        """
        if minio_client is None:
            logger.warning("visualization.upload.skipped", reason="MinIO client not configured")
            return None

        try:
            # Generate image bytes
            img_bytes = fig.to_image(format=format, engine="kaleido")
            stream = io.BytesIO(img_bytes)
            size = len(img_bytes)
            content_type = f"image/{format}"

            # Upload to MinIO
            minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=stream,
                length=size,
                content_type=content_type,
            )

            logger.info("visualization.upload.success", object_name=object_name)

            # Note: in production, return a pre-signed URL or the relative path
            # presigned_url = minio_client.presigned_get_object(self.bucket_name, object_name)
            return f"/api/storage/{self.bucket_name}/{object_name}"

        except Exception as e:
            logger.error("visualization.upload.failed", error=str(e), object_name=object_name)
            return None
