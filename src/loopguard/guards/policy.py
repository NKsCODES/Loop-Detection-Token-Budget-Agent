"""Guard policy: turn detector and budget signals into one verdict.

Verdicts:

* ``continue``  nothing to do.
* ``nudge``     add a short note for the model (soft budget warning) and carry on.
* ``pause``     stop and wait for a human decision (LangGraph ``interrupt``).
* ``abort``     end the run now with a clear stop reason.

Precedence: step limit, then loops, then hard budget, then soft budget. The step limit is
absolute; loops come before budget because a loop is the usual reason a budget runs out, and
the loop evidence is what a reviewer needs to see.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from loopguard.config import GuardConfig, ModelPricing
from loopguard.guards.budget import check_budget
from loopguard.guards.detectors import detect_loop
from loopguard.guards.fingerprint import ToolRecord

VerdictAction = Literal["continue", "nudge", "pause", "abort"]
VerdictKind = Literal["none", "loop", "budget", "step_limit", "soft_budget"]

LOOP_NUDGE_PREFIX = "[loopguard:loop]"
BUDGET_NUDGE_PREFIX = "[loopguard:budget]"


@dataclass(frozen=True)
class GuardVerdict:
    action: VerdictAction
    kind: VerdictKind = "none"
    detector: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.action == "abort"

    def allowed_actions(self) -> list[str]:
        if self.kind == "budget":
            return ["extend_budget", "reject"]
        return ["approve", "reject"]

    def event(self, step: int, **extra: Any) -> dict[str, Any]:
        return {
            "step": step,
            "kind": self.kind,
            "action": self.action,
            "detector": self.detector,
            "message": self.message,
            **extra,
        }


CONTINUE = GuardVerdict(action="continue")


def evaluate(
    *,
    records: Sequence[ToolRecord],
    steps: int,
    usage: dict[str, float],
    next_prompt_tokens: int,
    cfg: GuardConfig,
    pricing: ModelPricing,
    window_start: int = 0,
    extra_tokens: int = 0,
    soft_warned: bool = False,
) -> GuardVerdict:
    if not cfg.enabled:
        return CONTINUE

    if steps >= cfg.max_steps:
        return GuardVerdict(
            action=cfg.on_step_limit,
            kind="step_limit",
            detector="max_steps",
            message=f"Reached the step limit of {cfg.max_steps} tool rounds.",
            details={"steps": steps, "max_steps": cfg.max_steps},
        )

    if cfg.detect_loops:
        signal = detect_loop(records[window_start:], cfg)
        if signal is not None:
            return GuardVerdict(
                action=cfg.on_loop,
                kind="loop",
                detector=signal.detector,
                message=signal.message,
                details=signal.to_dict(),
            )

    if cfg.enforce_budget:
        budget = check_budget(
            usage, next_prompt_tokens, cfg, pricing, extra_tokens, already_warned=soft_warned
        )
        if budget is not None and budget.level == "hard":
            return GuardVerdict(
                action=cfg.on_budget,
                kind="budget",
                detector="token_budget",
                message=budget.message,
                details=budget.to_dict(),
            )
        if budget is not None and budget.level == "soft":
            return GuardVerdict(
                action="nudge",
                kind="soft_budget",
                detector="soft_budget",
                message=budget.message,
                details=budget.to_dict(),
            )

    return CONTINUE


def loop_nudge(message: str, note: str | None = None) -> str:
    text = (
        f"{LOOP_NUDGE_PREFIX} A reviewer let you continue, but you are repeating yourself: "
        f"{message} Do not repeat that call. Try a different approach, or give your best "
        "final answer now and say what you could not confirm."
    )
    if note:
        text += f" Reviewer note: {note}"
    return text


def budget_nudge(message: str) -> str:
    return (
        f"{BUDGET_NUDGE_PREFIX} {message} Finish with the information you already have "
        "unless one more tool call is essential."
    )
