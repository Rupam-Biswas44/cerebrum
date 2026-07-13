"""
Data Profiler Service

Uses Polars to efficiently read data files (CSV, Parquet, Excel, JSON)
in memory, infer schema, count missing values, and calculate basic statistics.
"""

from io import BytesIO
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


def profile_dataset(file_data: bytes, filename: str, content_type: str) -> dict[str, Any]:
    """
    Load data using Polars and generate a schema profile.
    
    Args:
        file_data: Raw bytes of the file.
        filename: Name of the file.
        content_type: MIME type of the file.
        
    Returns:
        Dictionary containing row_count, column_count, schema, and basic stats.
    """
    try:
        df = _load_dataframe(file_data, filename, content_type)
        
        row_count = df.height
        col_count = df.width
        
        # Infer schema and missing values
        schema_info = []
        for col_name in df.columns:
            dtype = str(df[col_name].dtype)
            null_count = df[col_name].null_count()
            
            col_info = {
                "name": col_name,
                "type": dtype,
                "null_count": null_count,
                "null_percentage": round((null_count / row_count) * 100, 2) if row_count > 0 else 0.0
            }
            
            # Basic stats for numeric columns
            if df[col_name].dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
                col_info["mean"] = df[col_name].mean()
                col_info["min"] = df[col_name].min()
                col_info["max"] = df[col_name].max()
            
            schema_info.append(col_info)
            
        return {
            "row_count": row_count,
            "column_count": col_count,
            "schema": schema_info,
        }
        
    except Exception as e:
        logger.error("data_profiler.failed", filename=filename, error=str(e))
        raise ValueError(f"Could not parse data file: {e}")


def _load_dataframe(file_data: bytes, filename: str, content_type: str) -> pl.DataFrame:
    """Helper to load bytes into a Polars DataFrame based on format."""
    data_io = BytesIO(file_data)
    
    if filename.endswith(".csv") or "csv" in content_type:
        return pl.read_csv(data_io, ignore_errors=True, infer_schema_length=1000)
    elif filename.endswith(".parquet") or "parquet" in content_type:
        return pl.read_parquet(data_io)
    elif filename.endswith(".json") or "json" in content_type:
        return pl.read_json(data_io)
    elif filename.endswith(".xlsx") or "spreadsheet" in content_type:
        return pl.read_excel(data_io)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, JSON, Parquet, or Excel.")
