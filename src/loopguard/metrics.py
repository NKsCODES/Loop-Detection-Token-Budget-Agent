"""Prometheus metrics.

Stuck rate in PromQL::

    sum(increase(loopguard_stuck_runs_total[1h]))
      / sum(increase(loopguard_runs_finished_total[1h]))
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

RUNS_STARTED = Counter("loopguard_runs_started_total", "Agent runs started", registry=REGISTRY)
RUNS_FINISHED = Counter(
    "loopguard_runs_finished_total",
    "Runs that reached a final state",
    ["status"],
    registry=REGISTRY,
)
STUCK_RUNS = Counter(
    "loopguard_stuck_runs_total", "Runs that ended at a hard limit", ["reason"], registry=REGISTRY
)
GUARD_TRIGGERS = Counter(
    "loopguard_guard_triggers_total",
    "Guard rule firings",
    ["kind", "detector", "outcome"],
    registry=REGISTRY,
)
HITL_PAUSES = Counter(
    "loopguard_hitl_pauses_total",
    "Runs paused for human review",
    ["kind", "detector"],
    registry=REGISTRY,
)
HITL_DECISIONS = Counter(
    "loopguard_hitl_decisions_total", "Reviewer decisions", ["action"], registry=REGISTRY
)
LLM_TOKENS = Counter("loopguard_llm_tokens_total", "LLM tokens", ["direction"], registry=REGISTRY)
LLM_COST = Counter("loopguard_llm_cost_usd_total", "Estimated LLM spend in USD", registry=REGISTRY)
RUN_STEPS = Histogram(
    "loopguard_run_steps",
    "Tool rounds per finished run",
    buckets=(1, 2, 3, 5, 8, 13, 20, 30),
    registry=REGISTRY,
)
RUN_TOKENS = Histogram(
    "loopguard_run_tokens",
    "Tokens per finished run",
    buckets=(500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000),
    registry=REGISTRY,
)
PENDING_REVIEWS = Gauge(
    "loopguard_pending_reviews", "Runs waiting for a reviewer", registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "loopguard_http_request_seconds",
    "HTTP request latency",
    ["method", "route", "status"],
    registry=REGISTRY,
)


def record_invocation(
    before: dict[str, Any],
    after: dict[str, Any],
    status: str,
    pending: dict[str, Any] | None,
    is_stuck: bool,
    stop_reason: str | None,
    steps: int,
) -> None:
    """Record metric deltas for one graph invocation.

    ``before`` and ``after`` are checkpointed state values. Working from state deltas, not from
    inside graph nodes, means a node that LangGraph re-runs on resume is never counted twice.
    """
    u0, u1 = before.get("usage", {}) or {}, after.get("usage", {}) or {}
    for key, direction in (("input_tokens", "input"), ("output_tokens", "output")):
        delta = u1.get(key, 0) - u0.get(key, 0)
        if delta > 0:
            LLM_TOKENS.labels(direction).inc(delta)
    cost = u1.get("cost_usd", 0.0) - u0.get("cost_usd", 0.0)
    if cost > 0:
        LLM_COST.inc(cost)

    new_events = (after.get("guard_events") or [])[len(before.get("guard_events") or []) :]
    for event in new_events:
        GUARD_TRIGGERS.labels(
            event.get("kind") or "",
            event.get("detector") or "",
            event.get("decision") or event.get("action") or "",
        ).inc()
        if event.get("decision"):
            HITL_DECISIONS.labels(event["decision"]).inc()

    if pending is not None:
        HITL_PAUSES.labels(pending.get("kind") or "", pending.get("detector") or "").inc()
        return
    RUNS_FINISHED.labels(status).inc()
    RUN_STEPS.observe(steps)
    RUN_TOKENS.observe(u1.get("total_tokens", 0))
    if is_stuck:
        STUCK_RUNS.labels(stop_reason or "unknown").inc()
