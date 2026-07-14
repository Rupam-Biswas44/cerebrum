"""
Tool Registry

A central registry where agents discover and invoke tools.
Tools are callable functions with typed schemas.

Architecture:
  - Tools are registered with @tool_registry.register(...)
  - Agents call tool_registry.execute(name, **kwargs) at runtime
  - LangGraph nodes can use list_tools() to build LLM tool schemas

This avoids hard-coding tool availability inside each agent and
lets us add/remove tools without modifying agent code.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ToolSpec:
    """Metadata and callable for a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        tags: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.tags = tags or []
        # Introspect the function signature to auto-generate schema
        sig = inspect.signature(fn)
        self.parameters = {
            param_name: {
                "type": str(param.annotation.__name__)
                if param.annotation != inspect.Parameter.empty
                else "any",
                "required": param.default == inspect.Parameter.empty,
            }
            for param_name, param in sig.parameters.items()
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to OpenAI function-calling schema format."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, info in self.parameters.items():
            properties[param_name] = {"type": info["type"], "description": ""}
            if info["required"]:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """
    Singleton registry for all agent tools.

    Usage:
        @tool_registry.register("run_sql", description="Execute SQL via DuckDB")
        async def run_sql(query: str, file_path: str) -> dict:
            ...
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a named tool."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = ToolSpec(name=name, description=description, fn=fn, tags=tags)
            self._tools[name] = spec
            logger.debug("tool.registered", name=name, tags=tags)
            return fn

        return decorator

    def get(self, name: str) -> ToolSpec:
        """Retrieve a tool spec by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry. Available: {list(self._tools)}")
        return self._tools[name]

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """
        Execute a registered tool by name with the given arguments.
        Handles both sync and async callables.
        """
        spec = self.get(name)
        logger.info("tool.execute.start", tool=name, kwargs=list(kwargs.keys()))
        try:
            if inspect.iscoroutinefunction(spec.fn):
                result = await spec.fn(**kwargs)
            else:
                result = spec.fn(**kwargs)
            logger.info("tool.execute.done", tool=name)
            return result
        except Exception as e:
            logger.error("tool.execute.failed", tool=name, error=str(e))
            raise

    def list_tools(self, tag: str | None = None) -> list[ToolSpec]:
        """List all registered tools, optionally filtered by tag."""
        tools = list(self._tools.values())
        if tag:
            tools = [t for t in tools if tag in t.tags]
        return tools

    def openai_schemas(self, tag: str | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all tools (or filtered by tag)."""
        return [t.to_openai_schema() for t in self.list_tools(tag=tag)]


# Global singleton — import this in agents and router files
tool_registry = ToolRegistry()


# ============================================================
# Built-in Tools Registration
# ============================================================


@tool_registry.register(
    "query_dataset",
    description="Execute a SQL query directly on a dataset file using DuckDB",
    tags=["data", "sql"],
)
async def query_dataset(file_path: str, query: str) -> dict[str, Any]:
    """Query a dataset file using the DuckDB query engine."""
    from services.query_engine import execute_sql_on_file

    result = execute_sql_on_file(file_path=file_path, query=query)
    return result.model_dump()


@tool_registry.register(
    "profile_dataset",
    description="Generate schema and statistical profile of a data file",
    tags=["data", "profiling"],
)
async def run_profile(file_path: str, filename: str) -> dict[str, Any]:
    """Profile a dataset file and return schema information."""
    import httpx

    # In production this would read from MinIO; for now read via HTTP presigned URL
    async with httpx.AsyncClient() as client:
        response = await client.get(file_path, timeout=30)
        response.raise_for_status()
        file_bytes = response.content

    from services.data_profiler import profile_dataset

    return profile_dataset(file_data=file_bytes, filename=filename, content_type="")


@tool_registry.register(
    "search_memory",
    description="Semantically search long-term agent memory for relevant context",
    tags=["memory", "search"],
)
async def search_memory_tool(project_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search agent memory using semantic vector search."""
    import uuid as _uuid

    from services.vector_store import semantic_search

    return await semantic_search(
        project_id=_uuid.UUID(project_id),
        query=query,
        top_k=top_k,
    )


@tool_registry.register(
    "store_insight",
    description="Store an agent-generated insight into long-term memory",
    tags=["memory", "write"],
)
async def store_insight(project_id: str, text: str, importance: float = 0.7) -> dict[str, str]:
    """Persist an insight to the vector store for future retrieval."""
    import uuid as _uuid

    from services.vector_store import VectorMemory, upsert_vectors

    mem = VectorMemory(
        id=str(_uuid.uuid4()),
        text=text,
        payload={"memory_type": "insight", "importance": importance},
    )
    await upsert_vectors(project_id=_uuid.UUID(project_id), memories=[mem])
    return {"status": "stored", "text_preview": text[:100]}
