"""
Analytics Service

Provides statistical profiling, outlier detection, correlation matrices,
and arbitrary SQL execution (via DuckDB) against Pandas DataFrames.
"""

from typing import Any

import duckdb
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class AnalyticsService:
    """Service for running analytical operations on datasets."""

    def detect_outliers(self, df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        Returns a dictionary mapping column name to outlier counts and bounds.
        """
        if columns is None:
            # Auto-select numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers_report = {}
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            outliers_report[col] = {
                "count": len(outliers),
                "percentage": round(len(outliers) / len(df) * 100, 2) if len(df) > 0 else 0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

        return outliers_report

    def calculate_correlations(self, df: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
        """
        Calculate the correlation matrix for all numeric columns.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}

        corr_matrix = numeric_df.corr(method=method)
        # Replace NaNs with None for JSON serialization
        corr_matrix = corr_matrix.replace({np.nan: None})

        return corr_matrix.to_dict()

    def execute_sql(self, df: pd.DataFrame, query: str) -> list[dict[str, Any]]:
        """
        Execute an arbitrary SQL query against the provided DataFrame.
        The dataframe is registered as a table named 'dataset'.
        """
        logger.info("analytics.sql.execute", query=query)
        try:
            # We use an in-memory DuckDB connection for fast SQL execution on Pandas DataFrames.
            con = duckdb.connect(database=":memory:")
            # Register the dataframe explicitly or let DuckDB automatically find it.
            # It's safer to register it as 'dataset'
            con.register("dataset", df)

            # Execute and fetch as dictionary
            result_df = con.execute(query).df()
            # Replace NaNs for JSON safety
            result_df = result_df.replace({np.nan: None})
            return result_df.to_dict(orient="records")
        except Exception as e:
            logger.error("analytics.sql.failed", query=query, error=str(e))
            raise ValueError(f"Failed to execute SQL: {e}") from e

    def generate_insights(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate a comprehensive set of automated insights for the dataframe.
        """
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "outliers": self.detect_outliers(df),
            "correlations": self.calculate_correlations(df),
        }
