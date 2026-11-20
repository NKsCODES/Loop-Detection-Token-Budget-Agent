"""LangGraph agent graphs.

Guarded graph::

    START -> agent --(tool calls)--> tools -> guard --(continue / nudge / approved)--> agent
               |                                 |
               +--(final answer)--> END          +--(abort / rejected)--> END
                                                 |
                                                 +--(pause) interrupt() waits for a reviewer

Baseline graph (for comparison): the textbook ReAct loop, ``agent <-> tools``, with only the
framework's recursion limit to stop it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from loopguard.agent.state import TERMINAL_STATUSES, AgentState
from loopguard.agent.tools import TOOL_CONTEXT, ToolContext
from loopguard.config import GuardConfig, ModelPricing
from loopguard.guards.fingerprint import ToolRecord, make_record
from loopguard.guards.policy import GuardVerdict, budget_nudge, evaluate, loop_nudge
from loopguard.logs import get_logger
from loopguard.tokens import TokenCounter, usage_delta

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an operations assistant for an Indian e-commerce company. Use the tools to "
    "answer. Amounts are in INR. Never repeat a tool call that already failed or already "
    "returned the same result. If a tool keeps failing or gives no new information, stop and "
    "explain what you could not confirm."
)

_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="loopguard-tool")


class ReviewDecision(BaseModel):
    """What a human reviewer sends to resume a paused run."""

    action: Literal["approve", "reject", "extend_budget"]
    extra_tokens: int | None = Field(None, ge=0, le=1_000_000)
    note: str | None = Field(None, max_length=1000)
    reviewer: str | None = Field(None, max_length=200)


def guard_config_from(config: RunnableConfig | None, default: GuardConfig) -> GuardConfig:
    raw = ((config or {}).get("configurable") or {}).get("guard")
    if raw is None:
        return default
    return raw if isinstance(raw, GuardConfig) else GuardConfig.model_validate(raw)


def _final_message(text: str) -> AIMessage:
    return AIMessage(content=text)


class _ToolRunner:
    """Executes tool calls with timeouts, error capture, output caps and fingerprinting."""

    def __init__(self, tools: Sequence[BaseTool], timeout_s: float, max_chars: int) -> None:
        self.tools = {t.name: t for t in tools}
        self.timeout_s = timeout_s
        self.max_chars = max_chars

    @staticmethod
    def _invoke(tool: BaseTool, args: dict[str, Any], ctx: ToolContext) -> str:
        token = TOOL_CONTEXT.set(ctx)
        try:
            return str(tool.invoke(args))
        finally:
            TOOL_CONTEXT.reset(token)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        history = [ToolRecord.from_dict(d) for d in state.get("tool_records", [])]
        seen = Counter(r.args_fp for r in history)
        step = int(state.get("steps", 0)) + 1
        messages: list[ToolMessage] = []
        records: list[dict[str, Any]] = []
        for i, call in enumerate(last.tool_calls):
            name, args = call["name"], call.get("args") or {}
            probe = make_record(step, name, args, "")
            ctx = ToolContext(run_call_index=len(history) + i, prior_same_calls=seen[probe.args_fp])
            seen[probe.args_fp] += 1
            tool = self.tools.get(name)
            is_error = False
            if tool is None:
                output, is_error = f"ERROR: unknown tool {name!r}.", True
            else:
                future = _EXECUTOR.submit(self._invoke, tool, args, ctx)
                try:
                    output = future.result(timeout=self.timeout_s)
                except FutureTimeout:
                    output, is_error = (
                        f"ERROR: tool {name} timed out after {self.timeout_s}s.",
                        True,
                    )
                except Exception as exc:  # tool raised: report to the model, do not crash
                    output, is_error = f"ERROR: {type(exc).__name__}: {exc}", True
            if len(output) > self.max_chars:
                output = (
                    output[: self.max_chars]
                    + f"\n...[truncated {len(output) - self.max_chars} chars]"
                )
            is_error = is_error or output.startswith("ERROR")
            messages.append(
                ToolMessage(
                    content=output,
                    tool_call_id=call["id"] or f"{name}-{step}-{i}",
                    name=name,
                    status="error" if is_error else "success",
                )
            )
            records.append(make_record(step, name, args, output, is_error).to_dict())
        return {"messages": messages, "tool_records": records, "steps": step}


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def _schema_tokens(tools: Sequence[BaseTool], counter: TokenCounter) -> int:
    return counter.count_text(json.dumps([convert_to_openai_tool(t) for t in tools]))


def build_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    guarded: bool = True,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    guard_config: GuardConfig | None = None,
    pricing: ModelPricing | None = None,
    counter: TokenCounter | None = None,
    tool_timeout_s: float = 20.0,
    max_tool_output_chars: int = 8_000,
    system_prompt: str = SYSTEM_PROMPT,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    default_cfg = guard_config or GuardConfig()
    pricing = pricing or ModelPricing()
    counter = counter or TokenCounter()
    bound = model.bind_tools(list(tools))
    schema_tokens = _schema_tokens(tools, counter)
    system = SystemMessage(content=system_prompt)

    def prompt_for(state: AgentState) -> list[BaseMessage]:
        return [system, *state.get("messages", [])]

    def agent(state: AgentState) -> dict[str, Any]:
        prompt = prompt_for(state)
        ai = bound.invoke(prompt)
        if not isinstance(ai, AIMessage):  # pragma: no cover - defensive
            ai = AIMessage(content=str(ai))
        return {"messages": [ai], "usage": usage_delta(ai, prompt, counter, pricing)}

    def guard(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        cfg = guard_config_from(config, default_cfg)
        records = [ToolRecord.from_dict(d) for d in state.get("tool_records", [])]
        steps = int(state.get("steps", 0))
        usage = state.get("usage", {}) or {}
        verdict = evaluate(
            records=records,
            steps=steps,
            usage=usage,
            next_prompt_tokens=counter.count_messages(prompt_for(state)) + schema_tokens,
            cfg=cfg,
            pricing=pricing,
            window_start=int(state.get("window_start", 0)),
            extra_tokens=int(state.get("extra_tokens", 0)),
            soft_warned=bool(state.get("soft_budget_warned", False)),
        )
        if verdict.action == "continue":
            return {}
        if verdict.action == "nudge":
            return {
                "messages": [HumanMessage(content=budget_nudge(verdict.message))],
                "soft_budget_warned": True,
                "guard_events": [verdict.event(steps)],
            }
        if verdict.action == "abort":
            return {
                "status": "aborted",
                "stop_reason": verdict.kind,
                "messages": [_final_message(f"Stopped by LoopGuard: {verdict.message}")],
                "guard_events": [verdict.event(steps)],
            }
        return _pause(state, verdict, steps, usage, records, cfg)

    def _pause(
        state: AgentState,
        verdict: GuardVerdict,
        steps: int,
        usage: dict[str, float],
        records: list[ToolRecord],
        cfg: GuardConfig,
    ) -> dict[str, Any]:
        # Everything above this line is a pure function of state, so re-running the node on
        # resume reaches the same verdict and interrupt() returns the reviewer's decision.
        raw = interrupt(
            {
                "kind": verdict.kind,
                "detector": verdict.detector,
                "message": verdict.message,
                "evidence": verdict.details.get("evidence", []),
                "details": {k: v for k, v in verdict.details.items() if k != "evidence"},
                "step": steps,
                "usage": usage,
                "allowed_actions": verdict.allowed_actions(),
            }
        )
        decision = raw if isinstance(raw, ReviewDecision) else ReviewDecision.model_validate(raw)
        action = decision.action
        if verdict.kind == "budget" and action == "approve":
            action = "extend_budget"
        if verdict.kind != "budget" and action == "extend_budget":
            action = "approve"
        event = verdict.event(
            steps, decision=action, reviewer=decision.reviewer, note=decision.note
        )
        if action == "reject":
            text = f"Stopped by reviewer: {verdict.message}"
            if decision.note:
                text += f" Note: {decision.note}"
            return {
                "status": "rejected",
                "stop_reason": verdict.kind,
                "messages": [_final_message(text)],
                "guard_events": [event],
            }
        if action == "extend_budget":
            extra = decision.extra_tokens
            if extra is None:
                extra = cfg.budget_extension_tokens
            event["extra_tokens"] = extra
            return {
                "extra_tokens": int(state.get("extra_tokens", 0)) + extra,
                "guard_events": [event],
            }
        # approve a loop: start a fresh detection window and tell the model what happened
        return {
            "window_start": len(records),
            "messages": [HumanMessage(content=loop_nudge(verdict.message, decision.note))],
            "guard_events": [event],
        }

    def route_after_guard(state: AgentState) -> str:
        return END if state.get("status") in TERMINAL_STATUSES else "agent"

    tool_runner = _ToolRunner(tools, tool_timeout_s, max_tool_output_chars)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_runner)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    if guarded:
        graph.add_node("guard", guard)
        graph.add_edge("tools", "guard")
        graph.add_conditional_edges("guard", route_after_guard, {"agent": "agent", END: END})
    else:
        graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
