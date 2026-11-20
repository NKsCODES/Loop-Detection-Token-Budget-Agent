"""LangGraph state for the guarded agent.

Every field the guard needs lives in the checkpointed state, never in process memory. A run
paused on one replica can be resumed on another.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from loopguard.tokens import merge_usage

TERMINAL_STATUSES = {"rejected", "aborted"}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_records: Annotated[list[dict[str, Any]], operator.add]  # ToolRecord.to_dict()
    guard_events: Annotated[list[dict[str, Any]], operator.add]
    usage: Annotated[dict[str, float], merge_usage]
    steps: int  # tool rounds executed
    window_start: int  # index into tool_records where loop detection starts
    extra_tokens: int  # budget extensions approved by reviewers
    soft_budget_warned: bool
    status: str  # set only when the guard ends the run: "rejected" or "aborted"
    stop_reason: str
