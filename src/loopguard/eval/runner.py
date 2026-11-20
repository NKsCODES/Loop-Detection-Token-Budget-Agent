"""Benchmark harness: run the 50 tasks under several agent variants and measure stuck rate.

Variants
--------
``baseline``         textbook LangGraph ReAct loop, no guard, recursion_limit=25.
``step_limit_only``  hard cap of 10 tool rounds, nothing else.
``loopguard``        full guard: step limit, loop detectors, token budget, human approval.

Human approval is played by ``SimulatedReviewer``, a fixed policy that mirrors a sensible
operator: give a looping agent one second chance with a hint, stop it on the second loop,
extend the budget once for a run that is making progress.

Metric definitions
------------------
* **Stuck**: the run ended at a hard limit (recursion limit, step limit or budget abort)
  instead of with an answer or a reviewer decision. Stuck rate = stuck tasks / 50.
* **Pass**: the final answer contains every expected string. For traps the expected answer
  is the honest one ("unavailable", "still running", ...).
* **Loop false positive**: a loop pause on a healthy task.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from loopguard.agent.graph import ReviewDecision, build_agent_graph
from loopguard.agent.runner import PAUSED, RunOutcome, invoke
from loopguard.agent.tools import default_tools
from loopguard.config import GuardConfig, ModelPricing
from loopguard.eval.tasks import TaskSpec, benchmark_tasks
from loopguard.tokens import TokenCounter

BASELINE_RECURSION_LIMIT = 25
MAX_REVIEWS_PER_RUN = 10


@dataclass(frozen=True)
class Variant:
    name: str
    guarded: bool
    guard: GuardConfig = field(default_factory=GuardConfig)
    recursion_limit: int | None = None

    def limit(self) -> int:
        return self.recursion_limit or self.guard.recursion_limit()


def default_variants() -> list[Variant]:
    return [
        Variant("baseline", guarded=False, recursion_limit=BASELINE_RECURSION_LIMIT),
        Variant(
            "step_limit_only",
            guarded=True,
            guard=GuardConfig(detect_loops=False, enforce_budget=False, max_steps=10),
        ),
        Variant("loopguard", guarded=True, guard=GuardConfig()),
    ]


def threshold_variants(values: Iterable[int] = (2, 3, 4)) -> list[Variant]:
    return [
        Variant(
            f"loopguard_n{n}",
            guarded=True,
            guard=GuardConfig(max_identical_calls=n, max_same_output=n, cycle_min_repeats=n),
        )
        for n in values
    ]


class SimulatedReviewer:
    """Deterministic stand-in for the human in the loop."""

    def __init__(self, max_budget_extensions: int = 1) -> None:
        self.max_budget_extensions = max_budget_extensions

    def decide(self, pending: dict[str, Any], events: Sequence[dict[str, Any]]) -> ReviewDecision:
        resolved_loops = sum(1 for e in events if e.get("kind") == "loop" and e.get("decision"))
        extensions = sum(1 for e in events if e.get("decision") == "extend_budget")
        if pending.get("kind") == "budget":
            if resolved_loops == 0 and extensions < self.max_budget_extensions:
                return ReviewDecision(action="extend_budget", reviewer="sim", note="progressing")
            return ReviewDecision(action="reject", reviewer="sim", note="budget spent on a loop")
        if resolved_loops == 0:
            return ReviewDecision(
                action="approve", reviewer="sim", note="Change approach or report what failed."
            )
        return ReviewDecision(action="reject", reviewer="sim", note="Looped again after a hint.")


@dataclass
class TaskResult:
    task_id: str
    category: str
    is_trap: bool
    variant: str
    status: str
    stop_reason: str | None
    is_stuck: bool
    passed: bool
    steps: int
    total_tokens: int
    cost_usd: float
    llm_calls: int
    reviews: list[dict[str, Any]]
    first_detector: str | None
    first_detection_step: int | None
    loop_false_positive: bool
    final_answer: str | None
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_task(
    task: TaskSpec,
    variant: Variant,
    graph: Any,
    reviewer: SimulatedReviewer,
) -> TaskResult:
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": f"{variant.name}:{task.id}",
            "guard": variant.guard.model_dump(),
        },
        "recursion_limit": variant.limit(),
    }
    t0 = time.perf_counter()
    outcome: RunOutcome = invoke(graph, {"messages": [HumanMessage(content=task.prompt)]}, config)
    reviews: list[dict[str, Any]] = []
    while outcome.status == PAUSED and outcome.pending and len(reviews) < MAX_REVIEWS_PER_RUN:
        decision = reviewer.decide(outcome.pending, outcome.guard_events)
        reviews.append(
            {
                "kind": outcome.pending.get("kind"),
                "detector": outcome.pending.get("detector"),
                "step": outcome.pending.get("step"),
                "decision": decision.action,
            }
        )
        outcome = invoke(graph, Command(resume=decision.model_dump()), config)

    triggers = [e for e in outcome.guard_events if e.get("kind") in ("loop", "step_limit")]
    first = triggers[0] if triggers else None
    loop_fp = not task.is_trap and any(r["kind"] == "loop" for r in reviews)
    usage = outcome.usage
    return TaskResult(
        task_id=task.id,
        category=task.category,
        is_trap=task.is_trap,
        variant=variant.name,
        status=outcome.status,
        stop_reason=outcome.stop_reason,
        is_stuck=outcome.is_stuck,
        passed=task.passed(outcome.final_answer),
        steps=outcome.steps,
        total_tokens=int(usage.get("total_tokens", 0)),
        cost_usd=float(usage.get("cost_usd", 0.0)),
        llm_calls=int(usage.get("llm_calls", 0)),
        reviews=reviews,
        first_detector=first.get("detector") if first else None,
        first_detection_step=first.get("step") if first else None,
        loop_false_positive=loop_fp,
        final_answer=outcome.final_answer,
        seconds=round(time.perf_counter() - t0, 4),
    )


def run_variant(
    variant: Variant,
    tasks: Sequence[TaskSpec],
    model: BaseChatModel,
    pricing: ModelPricing,
    reviewer: SimulatedReviewer | None = None,
) -> list[TaskResult]:
    graph = build_agent_graph(
        model,
        default_tools(),
        guarded=variant.guarded,
        checkpointer=InMemorySaver(),
        guard_config=variant.guard,
        pricing=pricing,
        counter=TokenCounter(use_tiktoken=False),
    )
    reviewer = reviewer or SimulatedReviewer()
    return [run_task(t, variant, graph, reviewer) for t in tasks]


def summarize(results: Sequence[TaskResult]) -> dict[str, Any]:
    n = len(results)
    traps = [r for r in results if r.is_trap]
    healthy = [r for r in results if not r.is_trap]
    detect_steps = [r.first_detection_step for r in traps if r.first_detection_step is not None]
    reviews = [rv for r in results for rv in r.reviews]
    return {
        "variant": results[0].variant if results else "",
        "tasks": n,
        "trap_tasks": len(traps),
        "stuck": sum(r.is_stuck for r in results),
        "stuck_rate": round(sum(r.is_stuck for r in results) / n, 4) if n else 0.0,
        "stuck_traps": sum(r.is_stuck for r in traps),
        "stuck_healthy": sum(r.is_stuck for r in healthy),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / n, 4) if n else 0.0,
        "healthy_passed": sum(r.passed for r in healthy),
        "traps_passed": sum(r.passed for r in traps),
        "traps_stopped_by_reviewer": sum(r.status == "rejected" for r in traps),
        "total_tokens": sum(r.total_tokens for r in results),
        "trap_tokens": sum(r.total_tokens for r in traps),
        "healthy_tokens": sum(r.total_tokens for r in healthy),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "llm_calls": sum(r.llm_calls for r in results),
        "reviews": len(reviews),
        "loop_reviews": sum(rv["kind"] == "loop" for rv in reviews),
        "budget_reviews": sum(rv["kind"] == "budget" for rv in reviews),
        "loop_false_positives": sum(r.loop_false_positive for r in results),
        "median_detection_step": statistics.median(detect_steps) if detect_steps else None,
        "detectors": dict(Counter(r.first_detector for r in traps if r.first_detector)),
        "status_counts": dict(Counter(r.status for r in results)),
    }


def by_category(results: Sequence[TaskResult]) -> list[dict[str, Any]]:
    rows = []
    for cat in sorted({r.category for r in results}):
        rs = [r for r in results if r.category == cat]
        rows.append(
            {
                "category": cat,
                "tasks": len(rs),
                "stuck": sum(r.is_stuck for r in rs),
                "passed": sum(r.passed for r in rs),
                "avg_tokens": round(statistics.mean(r.total_tokens for r in rs)),
                "avg_steps": round(statistics.mean(r.steps for r in rs), 1),
                "detectors": dict(Counter(r.first_detector for r in rs if r.first_detector)),
            }
        )
    return rows


def run_benchmark(
    model: BaseChatModel,
    variants: Sequence[Variant] | None = None,
    tasks: Sequence[TaskSpec] | None = None,
    pricing: ModelPricing | None = None,
) -> dict[str, Any]:
    tasks = list(tasks or benchmark_tasks())
    pricing = pricing or ModelPricing()
    out: dict[str, Any] = {"variants": {}, "results": []}
    for variant in variants or default_variants():
        results = run_variant(variant, tasks, model, pricing)
        out["variants"][variant.name] = {
            "summary": summarize(results),
            "by_category": by_category(results),
            "guard_config": variant.guard.model_dump() if variant.guarded else None,
            "recursion_limit": variant.limit(),
        }
        out["results"].extend(r.to_dict() for r in results)
    return out
