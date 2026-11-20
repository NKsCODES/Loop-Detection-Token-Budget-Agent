"""Run a compiled graph one invocation at a time and classify the outcome.

Shared by the API service and the evaluation harness, so the benchmark exercises the same
code path as production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from loopguard.logs import get_logger

log = get_logger(__name__)

# Outcome statuses
COMPLETED = "completed"  # the agent produced a final answer on its own
PAUSED = "paused"  # waiting for a reviewer
REJECTED = "rejected"  # a reviewer stopped the run
ABORTED = "aborted"  # the guard ended the run at a hard limit
STUCK = "stuck"  # the framework recursion limit fired (no guard)
FAILED = "failed"  # an exception escaped

HARD_LIMIT_REASONS = {"step_limit", "budget"}


@dataclass
class RunOutcome:
    status: str
    final_answer: str | None = None
    stop_reason: str | None = None
    pending: dict[str, Any] | None = None
    steps: int = 0
    usage: dict[str, float] = field(default_factory=dict)
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def is_stuck(self) -> bool:
        """Stuck = the run ended at a hard limit instead of an answer or a reviewer decision."""
        return self.status == STUCK or (
            self.status == ABORTED and self.stop_reason in HARD_LIMIT_REASONS
        )

    @property
    def is_final(self) -> bool:
        return self.status != PAUSED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_stuck"] = self.is_stuck
        return data


def _final_answer(values: dict[str, Any]) -> str | None:
    for msg in reversed(values.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return str(msg.content)
    return None


def invoke(
    graph: CompiledStateGraph[Any, Any, Any, Any], payload: Any, config: dict[str, Any]
) -> RunOutcome:
    error: str | None = None
    status: str | None = None
    rc = cast(RunnableConfig, config)
    try:
        graph.invoke(payload, rc)
    except GraphRecursionError as exc:
        status, error = STUCK, f"recursion limit {config.get('recursion_limit')} reached: {exc}"
    except Exception as exc:
        log.exception("run_failed", thread_id=config.get("configurable", {}).get("thread_id"))
        status, error = FAILED, f"{type(exc).__name__}: {exc}"

    snap = graph.get_state(rc)
    values: dict[str, Any] = dict(snap.values or {})
    pending = snap.interrupts[0].value if snap.interrupts else None
    if status is None:
        if pending is not None:
            status = PAUSED
        elif values.get("status") in (REJECTED, ABORTED):
            status = values["status"]
        else:
            status = COMPLETED
    return RunOutcome(
        status=status,
        final_answer=None if status in (PAUSED, STUCK, FAILED) else _final_answer(values),
        stop_reason=values.get("stop_reason") or ("recursion_limit" if status == STUCK else None),
        pending=pending,
        steps=int(values.get("steps", 0)),
        usage=dict(values.get("usage", {}) or {}),
        guard_events=list(values.get("guard_events", [])),
        tool_records=list(values.get("tool_records", [])),
        error=error,
    )
