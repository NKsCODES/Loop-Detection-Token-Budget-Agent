from loopguard.eval.runner import default_variants, run_benchmark, threshold_variants
from loopguard.eval.tasks import benchmark_tasks


def test_suite_shape():
    tasks = benchmark_tasks()
    assert len(tasks) == 50
    assert sum(t.is_trap for t in tasks) == 30


def test_benchmark_is_deterministic_and_guard_wins(model):
    a = run_benchmark(model, variants=default_variants())
    b = run_benchmark(model, variants=default_variants())
    sa = {k: v["summary"] for k, v in a["variants"].items()}
    sb = {k: v["summary"] for k, v in b["variants"].items()}
    assert sa == sb

    base, guard = sa["baseline"], sa["loopguard"]
    assert base["stuck"] == 30  # every trap loops until the recursion limit
    assert guard["stuck"] == 0
    assert guard["total_tokens"] < base["total_tokens"]
    assert guard["healthy_passed"] == 20
    assert guard["loop_false_positives"] <= 1


def test_threshold_two_has_more_false_positives(model):
    out = run_benchmark(model, variants=threshold_variants((2, 3)))
    n2 = out["variants"]["loopguard_n2"]["summary"]
    n3 = out["variants"]["loopguard_n3"]["summary"]
    assert n2["loop_false_positives"] > n3["loop_false_positives"]
