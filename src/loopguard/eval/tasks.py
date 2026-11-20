"""The 50-task stuck-rate benchmark.

20 healthy tasks and 30 loop traps (6 each of five trap types). Healthy tasks include the
cases most likely to cause false alarms: repeated calls to one tool with different
arguments, a job that only finishes on the third poll, a ten-lookup task over the per-tool
limit, and two long-document tasks that need a budget extension.

Each task carries a natural-language prompt, the strings a correct answer must contain, and
the scripted policy the offline model follows. Real models get only the prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from loopguard.agent.fake_llm import (
    UNKNOWN_TASK,
    Observation,
    PaginatePolicy,
    PingPongPolicy,
    PlanPolicy,
    Policy,
    RephrasePolicy,
    RetryPolicy,
)
from loopguard.agent.tools import order_subtotal

GST = 1.18


@dataclass(frozen=True)
class TaskSpec:
    id: str
    category: str
    prompt: str
    expected: tuple[str, ...]  # all must appear in the final answer (case-insensitive)
    policy: Policy

    @property
    def is_trap(self) -> bool:
        return self.category.startswith("trap_")

    def passed(self, answer: str | None) -> bool:
        text = (answer or "").lower()
        return bool(text) and all(e.lower() in text for e in self.expected)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _healthy() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []

    # Order total with GST (lookup + calculator)
    for oid in ("1001", "1004", "1005", "1008"):
        sub = order_subtotal(oid)
        total = round(sub * GST, 2)
        tid = f"gst-{oid}"
        tasks.append(
            TaskSpec(
                tid,
                "healthy_gst",
                f"What is the total for order {oid} including 18% GST?",
                (_fmt(total),),
                PlanPolicy(
                    tid,
                    final=f"Order {oid} total including 18% GST is INR {_fmt(total)}.",
                    plan=[
                        ("lookup_order", {"order_id": oid}),
                        ("calculator", {"expression": f"{sub} * {GST}"}),
                    ],
                ),
            )
        )

    # Combined subtotal: several calls to the same tool with different args
    for ids in (
        ("1007", "1008", "1009", "1010", "1011"),
        ("1002", "1004", "1006", "1008", "1010", "1012"),
        tuple(f"10{i:02d}" for i in range(1, 11)),
    ):
        total = sum(order_subtotal(i) for i in ids)
        tid = f"sum-{ids[0]}-{len(ids)}"
        plan = [("lookup_order", {"order_id": i}) for i in ids]
        plan.append(("calculator", {"expression": " + ".join(str(order_subtotal(i)) for i in ids)}))
        tasks.append(
            TaskSpec(
                tid,
                "healthy_multi_lookup",
                f"What is the combined subtotal of orders {', '.join(ids)}?",
                (_fmt(total),),
                PlanPolicy(tid, final=f"The combined subtotal is INR {_fmt(total)}.", plan=plan),
            )
        )

    for city, temp in (("Pune", "27 C"), ("Mumbai", "31 C")):
        tid = f"weather-{city.lower()}"
        tasks.append(
            TaskSpec(
                tid,
                "healthy_weather",
                f"What's the weather in {city} right now?",
                (temp,),
                PlanPolicy(
                    tid,
                    final=f"{city} is {temp} right now.",
                    plan=[("fetch_weather", {"city": city})],
                ),
            )
        )

    for tid, q, query, ans, exp in (
        (
            "kb-refund",
            "How many days do customers have to request a refund?",
            "refund policy",
            "Customers can request a refund within 30 days of delivery.",
            "30 days",
        ),
        (
            "kb-leave",
            "How many days of annual leave do full-time employees get?",
            "annual leave policy",
            "Full-time employees get 24 days of annual leave.",
            "24 days",
        ),
        (
            "kb-sla",
            "What is the first response time for a P1 incident?",
            "P1 SLA response time",
            "P1 incidents get a first response within 15 minutes.",
            "15 minutes",
        ),
    ):
        tasks.append(
            TaskSpec(
                tid,
                "healthy_kb",
                q,
                (exp,),
                PlanPolicy(tid, final=ans, plan=[("search_kb", {"query": query})]),
            )
        )

    # Legitimate polling: finishes on the third poll. A threshold of 2 would flag this.
    for jid in ("job-1001", "job-1002"):
        tid = f"poll-ok-{jid}"
        tasks.append(
            TaskSpec(
                tid,
                "healthy_polling",
                f"Is the nightly export {jid} finished? If yes, give me the result file.",
                (f"s3://reports/{jid}.csv",),
                RetryPolicy(
                    tid,
                    final=f"{jid} finished. Result file: s3://reports/{jid}.csv",
                    tool="check_job_status",
                    args={"job_id": jid},
                    done_marker="status=DONE",
                ),
            )
        )

    tid = "jobs-batch"
    tasks.append(
        TaskSpec(
            tid,
            "healthy_multi_lookup",
            "Check jobs job-1004, job-1005 and job-1006 and tell me which are done.",
            ("job-1004", "job-1005", "job-1006"),
            PlanPolicy(
                tid,
                final="job-1004, job-1005 and job-1006 are all done.",
                plan=[
                    ("check_job_status", {"job_id": j})
                    for j in ("job-1004", "job-1005", "job-1006")
                ],
            ),
        )
    )

    for name, mgr in (("Asha Patil", "Rohan Mehta"), ("Dev Kulkarni", "Sana Qureshi")):
        tid = f"org-{name.split()[0].lower()}"
        tasks.append(
            TaskSpec(
                tid,
                "healthy_org",
                f"Who is {name}'s manager?",
                (mgr,),
                PlanPolicy(
                    tid,
                    final=f"{name} reports to {mgr}.",
                    plan=[("get_manager", {"employee": name})],
                ),
            )
        )

    tid = "tickets-alpha"
    tasks.append(
        TaskSpec(
            tid,
            "healthy_pagination",
            "How many open tickets does project alpha have?",
            ("15",),
            PaginatePolicy(tid, final="Project alpha has 15 open tickets.", project="alpha"),
        )
    )

    # Long documents: legitimate work that needs a reviewer to extend the token budget
    for prefix, label in (("board-q", "board reports"), ("audit-2025-", "2025 audit reports")):
        tid = f"docs-{prefix.rstrip('-q')}"
        doc_ids = [f"{prefix}{i}" for i in range(1, 5)]
        tasks.append(
            TaskSpec(
                tid,
                "healthy_long_docs",
                f"Summarize the four {label} ({doc_ids[0]} to {doc_ids[-1]}) "
                "in three bullet points.",
                ("western region", "logistics"),
                PlanPolicy(
                    tid,
                    final=(
                        f"Summary of {label}: steady growth in the western region; "
                        "higher logistics cost in the festive season; a hiring freeze in support."
                    ),
                    plan=[("read_document", {"doc_id": d}) for d in doc_ids],
                ),
            )
        )
    return tasks


def _traps() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []

    for city in ("Chennai", "Kolkata", "Jaipur", "Ahmedabad", "Kochi", "Surat"):
        tid = f"trap-503-{city.lower()}"
        tasks.append(
            TaskSpec(
                tid,
                "trap_retry_error",
                f"What's the weather in {city} right now?",
                ("unavailable",),
                RetryPolicy(
                    tid,
                    final="",
                    tool="fetch_weather",
                    args={"city": city},
                    done_marker="humidity",
                    graceful=(
                        f"The weather service is unavailable for {city} right now "
                        "(HTTP 503 on every try). Please check again later."
                    ),
                ),
            )
        )

    for n in range(1, 7):
        jid = f"job-200{n}"
        tid = f"trap-poll-{jid}"
        tasks.append(
            TaskSpec(
                tid,
                "trap_polling",
                f"Is the nightly export {jid} finished? If yes, give me the result file.",
                ("still running",),
                RetryPolicy(
                    tid,
                    final="",
                    tool="check_job_status",
                    args={"job_id": jid},
                    done_marker="status=DONE",
                    graceful=(
                        f"{jid} is still running with no progress reported, so I "
                        "cannot give you a result file yet."
                    ),
                ),
            )
        )

    searches = {
        "parental leave in the Antarctica office": [
            "parental leave Antarctica office",
            "Antarctica parental leave rules",
            "maternity paternity leave Antarctica",
            "Antarctica staff parental benefits",
            "parental leave remote research station",
        ],
        "crypto salary payout": [
            "crypto salary payout",
            "salary paid in bitcoin",
            "cryptocurrency payroll option",
            "pay salary in crypto tokens",
            "bitcoin compensation rules",
        ],
        "pet insurance benefit": [
            "pet insurance benefit",
            "pet insurance cover for staff",
            "dog cat insurance plan",
            "employee pet health plan",
            "pet care benefits",
        ],
        "four-day work week pilot": [
            "four-day work week pilot",
            "compressed schedule pilot",
            "four day working week",
            "shorter work week program",
            "32 hour week trial",
        ],
        "sabbatical after ten years": [
            "sabbatical after ten years",
            "long service sabbatical",
            "career break policy",
            "paid sabbatical rules",
            "extended unpaid break",
        ],
        "gym membership reimbursement": [
            "gym membership reimbursement",
            "fitness allowance",
            "gym fees refund for staff",
            "wellness stipend",
            "health club membership",
        ],
    }
    for topic, queries in searches.items():
        tid = f"trap-search-{topic.split()[0].lower()}"
        tasks.append(
            TaskSpec(
                tid,
                "trap_empty_search",
                f"What is our company policy on {topic}?",
                ("no information",),
                RephrasePolicy(
                    tid,
                    final="",
                    queries=queries,
                    graceful=(
                        f"I found no information about {topic} in the knowledge "
                        "base. Please check with HR directly."
                    ),
                ),
            )
        )

    for name in (
        "Vikram Singh",
        "Neha Joshi",
        "Arjun Rao",
        "Priya Nair",
        "Karan Shah",
        "Meera Iyer",
    ):
        tid = f"trap-pingpong-{name.split()[0].lower()}"
        tasks.append(
            TaskSpec(
                tid,
                "trap_ping_pong",
                f"Who is {name}'s manager?",
                ("could not determine",),
                PingPongPolicy(
                    tid,
                    final="",
                    employee=name,
                    graceful=(
                        f"I could not determine {name}'s manager: the org chart "
                        "and the HR record each point to the other."
                    ),
                ),
            )
        )

    for proj in (
        "legacy-crm",
        "legacy-billing",
        "legacy-hr",
        "legacy-erp",
        "legacy-wms",
        "legacy-pos",
    ):
        tid = f"trap-pages-{proj}"
        tasks.append(
            TaskSpec(
                tid,
                "trap_runaway_pagination",
                f"How many open tickets does project {proj} have?",
                ("pagination",),
                PaginatePolicy(
                    tid,
                    final="",
                    project=proj,
                    graceful=(
                        f"The ticket API for {proj} keeps returning another page "
                        "(the pagination never ends), so I cannot give a reliable "
                        "count."
                    ),
                ),
            )
        )
    return tasks


@lru_cache
def benchmark_tasks() -> tuple[TaskSpec, ...]:
    tasks = _healthy() + _traps()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    return tuple(tasks)


def default_resolver() -> Callable[[Observation], Policy]:
    by_prompt = {t.prompt: t.policy for t in benchmark_tasks()}

    def resolve(obs: Observation) -> Policy:
        return by_prompt.get(obs.task.strip(), UNKNOWN_TASK)

    return resolve
