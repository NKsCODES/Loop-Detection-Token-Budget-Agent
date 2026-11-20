from loopguard.config import GuardConfig, ModelPricing
from loopguard.guards.budget import check_budget
from loopguard.guards.fingerprint import make_record
from loopguard.guards.policy import evaluate

PRICING = ModelPricing()


def test_budget_ok():
    assert check_budget({"total_tokens": 1000}, 500, GuardConfig(), PRICING) is None


def test_soft_then_hard():
    cfg = GuardConfig(token_budget=10_000, cost_budget_usd=None)
    soft = check_budget({"total_tokens": 7_500}, 600, cfg, PRICING)
    assert soft and soft.level == "soft"
    assert check_budget({"total_tokens": 7_500}, 600, cfg, PRICING, already_warned=True) is None
    hard = check_budget({"total_tokens": 9_000}, 1_500, cfg, PRICING)
    assert hard and hard.level == "hard" and hard.projected_tokens == 10_500


def test_extension_raises_limit():
    cfg = GuardConfig(token_budget=10_000, cost_budget_usd=None)
    assert (
        check_budget(
            {"total_tokens": 9_000}, 1_500, cfg, PRICING, extra_tokens=5_000, already_warned=True
        )
        is None
    )


def test_cost_cap():
    cfg = GuardConfig(token_budget=1_000_000, cost_budget_usd=0.01)
    sig = check_budget({"total_tokens": 1000, "cost_usd": 0.009}, 1000, cfg, PRICING)
    assert sig and sig.level == "hard" and "cost" in sig.message


def _eval(records, steps, cfg, usage=None, window_start=0):
    return evaluate(
        records=records,
        steps=steps,
        usage=usage or {},
        next_prompt_tokens=100,
        cfg=cfg,
        pricing=PRICING,
        window_start=window_start,
    )


def test_step_limit_aborts():
    v = _eval([], steps=20, cfg=GuardConfig(max_steps=20))
    assert v.action == "abort" and v.kind == "step_limit"


def test_loop_pauses_and_window_resets():
    records = [make_record(i, "fetch_weather", {"city": "Kochi"}, "ERROR 503") for i in range(3)]
    v = _eval(records, steps=3, cfg=GuardConfig())
    assert v.action == "pause" and v.kind == "loop" and v.allowed_actions() == ["approve", "reject"]
    assert _eval(records, steps=3, cfg=GuardConfig(), window_start=3).action == "continue"


def test_disabled_guard_continues():
    records = [make_record(i, "fetch_weather", {"city": "Kochi"}, "ERROR 503") for i in range(9)]
    assert _eval(records, steps=50, cfg=GuardConfig(enabled=False)).action == "continue"


def test_budget_pause_offers_extension():
    v = _eval([], steps=1, cfg=GuardConfig(token_budget=1_000), usage={"total_tokens": 990})
    assert v.kind == "budget" and v.allowed_actions() == ["extend_budget", "reject"]
