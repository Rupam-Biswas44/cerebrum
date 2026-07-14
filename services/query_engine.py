"""
In-Memory Query Engine Service

Uses DuckDB to execute fast analytical SQL queries directly on files
stored in MinIO (Parquet, CSV) without loading them entirely into memory.
"""

from typing import Any

import duckdb
import structlog
from pydantic import BaseModel

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    execution_time_ms: float
    row_count: int


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Initializes an in-memory DuckDB connection configured with the MinIO
    credentials for querying data directly from the object store.
    """
    conn = duckdb.connect(database=":memory:")

    # Configure S3/MinIO extension
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

    conn.execute(f"SET s3_endpoint='{settings.MINIO_HOST}:{settings.MINIO_PORT}';")
    conn.execute(f"SET s3_access_key_id='{settings.MINIO_ROOT_USER}';")
    conn.execute(f"SET s3_secret_access_key='{settings.MINIO_ROOT_PASSWORD}';")
    conn.execute("SET s3_use_ssl=false;")  # Change to true in production with TLS
    conn.execute("SET s3_url_style='path';")

    return conn


def execute_sql_on_file(file_path: str, query: str) -> QueryResult:
    """
    Executes a SQL query against a remote file in MinIO using DuckDB.
    The file_path should be in the format 's3://bucket-name/path/to/file.csv'

    Args:
        file_path: S3 URI to the dataset.
        query: The SQL query (must select from 'dataset').

    Returns:
        QueryResult containing columns, rows, and timing.
    """
    import time

    start_time = time.time()

    # We replace a virtual table name 'dataset' with the actual S3 file read function
    # e.g., SELECT * FROM dataset -> SELECT * FROM read_csv_auto('s3://...')
    if file_path.endswith(".parquet"):
        read_stmt = f"read_parquet('{file_path}')"
    elif file_path.endswith(".csv"):
        read_stmt = f"read_csv_auto('{file_path}')"
    elif file_path.endswith(".json"):
        read_stmt = f"read_json_auto('{file_path}')"
    else:
        raise ValueError("Unsupported file format for direct SQL query.")

    # Inject the actual file reader into the query
    # (In a real scenario, we'd use a more robust SQL parser/AST modifier)
    executable_query = query.replace("FROM dataset", f"FROM {read_stmt}")

    try:
        conn = get_duckdb_connection()
        result = conn.execute(executable_query).df()

        # Convert pandas dataframe to list of dicts
        records = result.to_dict(orient="records")
        columns = list(result.columns)

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            columns=columns,
            rows=records,
            execution_time_ms=round(execution_time, 2),
            row_count=len(records),
        )

    except duckdb.Error as e:
        logger.error("duckdb.query.failed", file_path=file_path, error=str(e))
        raise ValueError(f"SQL execution failed: {e}") from e
    finally:
        if "conn" in locals():
            conn.close()
