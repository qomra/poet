"""
Session and conversation models — used by the agent harness and API.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class ToolCall(BaseModel):
    id: str                        # tool_use_id from the LLM
    name: str                      # tool name
    input: dict[str, Any]          # tool arguments


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    content: str                   # JSON-serialised result
    is_error: bool = False


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str | None = None
    language: str = "ar"           # ar | en
    tradition_context: str | None = None  # fusha | nabati | modern
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def history_for_llm(self) -> list[dict[str, Any]]:
        """Serialize messages into the format expected by LLM providers."""
        result = []
        for m in self.messages:
            if m.role == Role.SYSTEM:
                continue  # system prompt handled separately
            if m.role in (Role.USER, Role.ASSISTANT) and m.content:
                result.append({"role": m.role, "content": m.content})
            if m.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in m.tool_calls
                    ],
                })
            if m.tool_results:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                            **({"is_error": True} if tr.is_error else {}),
                        }
                        for tr in m.tool_results
                    ],
                })
        return result
