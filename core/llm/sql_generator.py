"""
SQL Generator

Uses LangChain and an LLM to convert a natural language question into
a SQL query that can be executed against DuckDB.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SQLGenerationResponse(BaseModel):
    query: str = Field(..., description="The generated SQL query.")
    reasoning: str = Field(..., description="Explanation of how the query works.")


SQL_GENERATOR_PROMPT = """You are an expert Data Analyst and SQL developer.
Your job is to translate a user's natural language question into a valid SQL query.

The query will be executed using DuckDB against a table named "dataset".

IMPORTANT RULES:
1. ONLY return a read-only SELECT statement. No INSERT, UPDATE, DELETE, or DROP.
2. Use standard SQL syntax compatible with DuckDB.
3. The table name to select from is ALWAYS "dataset".
4. Refer to the provided schema to ensure you only select columns that exist.
5. If the question cannot be answered with the given schema, output a query that returns 0 rows
   (e.g., SELECT * FROM dataset WHERE 1=0).

Output strictly in JSON format as specified.
"""


async def generate_sql_from_nl(
    question: str, schema: dict | str, llm_model: str = "gpt-4o", temperature: float = 0.0
) -> SQLGenerationResponse:
    """
    Generate a SQL query from a natural language question and a dataset schema.
    """
    logger.info("llm.sql_generator.start", question=question)

    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    # Bind the LLM to strictly output our Pydantic schema
    structured_llm = llm.with_structured_output(SQLGenerationResponse)

    schema_str = str(schema)
    user_message = f"Schema of table 'dataset':\n{schema_str}\n\nUser Question: {question}"

    messages = [
        SystemMessage(content=SQL_GENERATOR_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        raw_response = await structured_llm.ainvoke(messages)
        # LangChain may return a dict if pydantic binding isn't perfect
        if isinstance(raw_response, dict):
            response = SQLGenerationResponse(**raw_response)
        else:
            response = raw_response

        # Additional safety check: ensure the query is a SELECT
        cleaned_query = response.query.strip().upper()
        if not cleaned_query.startswith("SELECT"):
            logger.warning("llm.sql_generator.unsafe_query", query=response.query)
            # Override with a safe default
            response.query = "SELECT * FROM dataset LIMIT 0;"
            response.reasoning = "Query blocked for safety reasons (not a SELECT)."

        logger.info("llm.sql_generator.complete", query=response.query)
        return response

    except Exception as e:
        logger.error("llm.sql_generator.failed", error=str(e))
        return SQLGenerationResponse(
            query="SELECT * FROM dataset LIMIT 0;", reasoning=f"Failed to generate query: {str(e)}"
        )
