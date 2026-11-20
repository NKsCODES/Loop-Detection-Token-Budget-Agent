"""Token and cost budget checks.

The hard check is a preflight: it runs before the next LLM call and asks "will the next
prompt push this run over budget?". Stopping before the spend is cheaper than noticing after.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from loopguard.config import GuardConfig, ModelPricing


@dataclass(frozen=True)
class BudgetSignal:
    level: str  # "soft" or "hard"
    used_tokens: int
    projected_tokens: int
    limit_tokens: int
    used_cost_usd: float
    limit_cost_usd: float | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_budget(
    usage: dict[str, float],
    next_prompt_tokens: int,
    cfg: GuardConfig,
    pricing: ModelPricing,
    extra_tokens: int = 0,
    already_warned: bool = False,
) -> BudgetSignal | None:
    used = int(usage.get("total_tokens", 0))
    cost = float(usage.get("cost_usd", 0.0))
    limit = cfg.token_budget + extra_tokens
    projected = used + next_prompt_tokens
    projected_cost = cost + pricing.cost(next_prompt_tokens, 0)

    # Cost cap scales with any approved token extension, at the input price.
    cost_limit = None
    if cfg.cost_budget_usd is not None:
        cost_limit = cfg.cost_budget_usd + pricing.cost(extra_tokens, 0)

    over_tokens = projected > limit
    over_cost = cost_limit is not None and projected_cost > cost_limit
    if over_tokens or over_cost:
        what = "token" if over_tokens else "cost"
        return BudgetSignal(
            level="hard",
            used_tokens=used,
            projected_tokens=projected,
            limit_tokens=limit,
            used_cost_usd=round(cost, 6),
            limit_cost_usd=cost_limit,
            message=(
                f"The next call would exceed the {what} budget "
                f"({projected:,} of {limit:,} tokens, ${projected_cost:.4f})."
            ),
        )

    if not already_warned and projected >= cfg.soft_budget_ratio * limit:
        return BudgetSignal(
            level="soft",
            used_tokens=used,
            projected_tokens=projected,
            limit_tokens=limit,
            used_cost_usd=round(cost, 6),
            limit_cost_usd=cost_limit,
            message=f"{projected / limit:.0%} of the token budget is used.",
        )
    return None
