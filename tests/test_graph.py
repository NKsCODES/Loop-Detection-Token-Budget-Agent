from langgraph.types import Command

from loopguard.agent.runner import ABORTED, COMPLETED, PAUSED, REJECTED, STUCK, invoke
from loopguard.config import GuardConfig
from tests.conftest import config_for, start_payload


def test_healthy_task_completes_without_review(make_graph, tasks):
    t = tasks["gst-1001"]
    out = invoke(make_graph(), start_payload(t.prompt), config_for("h1"))
    assert out.status == COMPLETED and t.passed(out.final_answer)
    assert out.steps == 2 and out.usage["llm_calls"] == 3


def test_baseline_gets_stuck_on_trap(make_graph, tasks):
    t = tasks["trap-503-chennai"]
    out = invoke(make_graph(guarded=False), start_payload(t.prompt), config_for("b1", limit=25))
    assert out.status == STUCK and out.is_stuck


def test_guard_pauses_then_recovers_on_approve(make_graph, tasks):
    t = tasks["trap-503-kolkata"]  # this seed recovers after one hint
    g, cfg = make_graph(), config_for("g1")
    out = invoke(g, start_payload(t.prompt), cfg)
    assert out.status == PAUSED and out.pending["detector"] == "exact_repeat"
    assert out.steps == 3 and len(out.pending["evidence"]) == 3
    out = invoke(g, Command(resume={"action": "approve", "reviewer": "t"}), cfg)
    assert out.status == COMPLETED and t.passed(out.final_answer)
    assert out.guard_events[-1]["decision"] == "approve"


def test_reject_stops_the_run(make_graph, tasks):
    t = tasks["trap-poll-job-2001"]
    g, cfg = make_graph(), config_for("g2")
    assert invoke(g, start_payload(t.prompt), cfg).status == PAUSED
    out = invoke(g, Command(resume={"action": "reject", "note": "job is hung"}), cfg)
    assert out.status == REJECTED and not out.is_stuck
    assert "job is hung" in out.final_answer


def test_budget_pause_and_extension(make_graph, tasks):
    t = tasks["docs-board"]
    g, cfg = make_graph(), config_for("g3")
    out = invoke(g, start_payload(t.prompt), cfg)
    assert out.status == PAUSED and out.pending["kind"] == "budget"
    out = invoke(g, Command(resume={"action": "extend_budget", "extra_tokens": 10_000}), cfg)
    assert out.status == COMPLETED and t.passed(out.final_answer)


def test_step_limit_abort_counts_as_stuck(make_graph, tasks):
    t = tasks["trap-pages-legacy-crm"]
    gc = GuardConfig(detect_loops=False, enforce_budget=False, max_steps=5)
    out = invoke(make_graph(cfg=gc), start_payload(t.prompt), config_for("g4", gc))
    assert out.status == ABORTED and out.stop_reason == "step_limit" and out.is_stuck
    assert out.steps == 5


def test_legit_polling_is_not_flagged(make_graph, tasks):
    t = tasks["poll-ok-job-1001"]
    out = invoke(make_graph(), start_payload(t.prompt), config_for("g5"))
    assert out.status == COMPLETED and out.guard_events == []


def test_per_run_override_via_config(make_graph, tasks):
    t = tasks["trap-503-chennai"]
    gc = GuardConfig(on_loop="abort")
    out = invoke(make_graph(), start_payload(t.prompt), config_for("g6", gc))
    assert out.status == ABORTED and out.stop_reason == "loop" and not out.is_stuck


def test_tool_errors_do_not_crash(make_graph):
    from langchain_core.messages import AIMessage

    from loopguard.agent.graph import _ToolRunner
    from loopguard.agent.tools import default_tools

    runner = _ToolRunner(default_tools(), timeout_s=5, max_chars=50)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "nope", "args": {}, "id": "1", "type": "tool_call"},
                    {
                        "name": "calculator",
                        "args": {"expression": "__import__('os')"},
                        "id": "2",
                        "type": "tool_call",
                    },
                    {
                        "name": "read_document",
                        "args": {"doc_id": "board-q1"},
                        "id": "3",
                        "type": "tool_call",
                    },
                ],
            )
        ]
    }
    out = runner(state)
    msgs = out["messages"]
    assert msgs[0].status == "error" and "unknown tool" in msgs[0].content
    assert msgs[1].content.startswith("ERROR")
    assert "truncated" in msgs[2].content
