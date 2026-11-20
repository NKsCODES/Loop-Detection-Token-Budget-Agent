from loopguard.config import GuardConfig
from loopguard.guards.detectors import (
    detect_cycle,
    detect_exact_repeat,
    detect_loop,
    detect_stagnation,
    detect_tool_overuse,
)
from loopguard.guards.fingerprint import make_record

CFG = GuardConfig()


def recs(calls):
    return [make_record(i + 1, t, a, o) for i, (t, a, o) in enumerate(calls)]


def test_exact_repeat_fires_at_threshold():
    two = recs([("check_job_status", {"job_id": "job-2001"}, "RUNNING")] * 2)
    assert detect_exact_repeat(two, CFG) is None
    three = recs([("check_job_status", {"job_id": "job-2001"}, "RUNNING")] * 3)
    sig = detect_exact_repeat(three, CFG)
    assert sig and sig.detector == "exact_repeat" and sig.count == 3


def test_same_input_different_output_is_not_exact_repeat():
    r = recs(
        [
            ("check_job_status", {"job_id": "job-1001"}, s)
            for s in ("RUNNING 10%", "RUNNING 50%", "DONE")
        ]
    )
    assert detect_exact_repeat(r, CFG) is None


def test_cycle_detects_ping_pong_with_changing_nonce():
    calls = []
    for i in range(6):
        tool = "get_manager" if i % 2 == 0 else "get_hr_record"
        calls.append((tool, {"employee": "Neha Joshi"}, f"see other record (ref {4100 + i})"))
    r = recs(calls)
    assert detect_exact_repeat(r, CFG) is None  # nonce defeats exact matching
    sig = detect_cycle(r, CFG)
    assert sig and sig.detector == "cycle"


def test_stagnation_on_rephrased_search():
    qs = ["pet insurance", "pet insurance cover", "dog cat insurance plan"]
    r = recs([("search_kb", {"query": q}, f'No results found for "{q}".') for q in qs])
    sig = detect_stagnation(r, CFG)
    assert sig and sig.detector == "stagnation" and sig.count == 3


def test_tool_overuse_backstop():
    r = recs(
        [("list_tickets", {"project": "legacy-crm", "page": p}, f"page {p}") for p in range(1, 9)]
    )
    assert detect_exact_repeat(r, CFG) is None
    assert detect_stagnation(r, CFG) is None
    assert detect_cycle(r, CFG) is None
    sig = detect_tool_overuse(r, CFG)
    assert sig and sig.count == 8


def test_healthy_distinct_lookups_do_not_trigger():
    r = recs(
        [
            ("lookup_order", {"order_id": str(1000 + i)}, f"order {i} total {i * 10}")
            for i in range(1, 7)
        ]
    )
    assert detect_loop(r, CFG) is None


def test_priority_exact_repeat_first():
    r = recs([("fetch_weather", {"city": "Chennai"}, "ERROR 503")] * 8)
    sig = detect_loop(r, CFG)
    assert sig and sig.detector == "exact_repeat"
