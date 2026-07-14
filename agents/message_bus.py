"""
Agent Message Bus

Enables agents to communicate with each other during multi-agent
orchestration, passing typed messages through a Redis pub/sub channel.

This is the backbone of Cerebrum's multi-agent architecture:
  - Agents publish results and requests to named channels
  - Downstream agents subscribe and react to messages
  - The orchestrator uses this to coordinate parallel execution

Message Types:
  - AGENT_STARTED: An agent has begun execution
  - AGENT_COMPLETED: An agent has returned a result
  - AGENT_FAILED: An agent raised an error
  - HANDOFF: An agent is passing control to another agent
  - TOOL_CALL: An agent is requesting a tool to be executed
  - TOOL_RESULT: A tool has returned its result
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    HANDOFF = "handoff"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS_UPDATE = "status_update"


class AgentMessage:
    """A typed message passed between agents via the message bus."""

    def __init__(
        self,
        message_type: MessageType,
        sender_agent: str,
        task_id: str,
        payload: dict[str, Any],
        target_agent: str | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.message_type = message_type
        self.sender_agent = sender_agent
        self.target_agent = target_agent
        self.task_id = task_id
        self.payload = payload
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "message_type": self.message_type.value,
                "sender_agent": self.sender_agent,
                "target_agent": self.target_agent,
                "task_id": self.task_id,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> AgentMessage:
        d = json.loads(data)
        msg = cls.__new__(cls)
        msg.id = d["id"]
        msg.message_type = MessageType(d["message_type"])
        msg.sender_agent = d["sender_agent"]
        msg.target_agent = d.get("target_agent")
        msg.task_id = d["task_id"]
        msg.payload = d["payload"]
        msg.timestamp = d["timestamp"]
        return msg


def _channel_name(task_id: str) -> str:
    """Generate the Redis pub/sub channel name for a given task."""
    return f"cerebrum:task:{task_id}:messages"


async def publish_message(redis: Any, message: AgentMessage) -> None:
    """
    Publish an agent message to the task's Redis pub/sub channel.

    Args:
        redis: Active aioredis client.
        message: The AgentMessage to publish.
    """
    channel = _channel_name(message.task_id)
    await redis.publish(channel, message.to_json())
    logger.debug(
        "message_bus.published",
        type=message.message_type.value,
        sender=message.sender_agent,
        task_id=message.task_id,
    )

    # Also persist last N messages to a Redis list for replay/debugging
    history_key = f"cerebrum:task:{message.task_id}:history"
    await redis.lpush(history_key, message.to_json())
    await redis.ltrim(history_key, 0, 99)  # Keep last 100 messages
    await redis.expire(history_key, 86400)  # Expire after 24 hours


async def get_task_message_history(redis: Any, task_id: str) -> list[AgentMessage]:
    """
    Retrieve the full message history for a task from Redis.
    Useful for debugging and building audit trails.
    """
    history_key = f"cerebrum:task:{task_id}:history"
    raw_messages = await redis.lrange(history_key, 0, -1)
    messages = []
    for raw in reversed(raw_messages):  # oldest first
        try:
            messages.append(AgentMessage.from_json(raw))
        except Exception as e:  # noqa: BLE001
            logger.warning("message_bus.parse.failed", error=str(e))
    return messages


async def broadcast_status(
    redis: Any,
    task_id: str,
    agent_name: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Convenience wrapper to broadcast an agent status update.

    Args:
        redis: Active aioredis client.
        task_id: The running task ID.
        agent_name: Name of the agent broadcasting the status.
        status: Human-readable status string.
        details: Optional extra context.
    """
    msg = AgentMessage(
        message_type=MessageType.STATUS_UPDATE,
        sender_agent=agent_name,
        task_id=task_id,
        payload={"status": status, **(details or {})},
    )
    await publish_message(redis, msg)
