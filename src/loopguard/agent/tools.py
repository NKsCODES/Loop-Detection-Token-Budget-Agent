"""Tool environment for the operations agent.

The tools model a small Indian e-commerce company: orders, weather for delivery cities, a
batch-job scheduler, a policy knowledge base, an org chart, a ticket tracker and long reports.

Some inputs are deliberate loop traps, each matching a failure mode seen in production agents:

=====================  ==============================================  ======================
Trap                   What the tool does                              Typical agent reaction
=====================  ==============================================  ======================
Upstream outage        ``fetch_weather`` returns HTTP 503 every time   retry the same call
Never-finishing job    ``check_job_status`` always says RUNNING        poll forever
Missing knowledge      ``search_kb`` finds nothing                     rephrase forever
Circular records       manager and HR record point at each other       ping-pong
Broken pagination      ``list_tickets`` always has a next page         page forever
=====================  ==============================================  ======================

Outputs carry realistic noise (request ids, timestamps, nonces, echoed queries) so the
detector has to normalise before it can see a repeat.

Tools are deterministic. Per-run state, such as "how many times was this job polled", comes
from a ``ToolContext`` that the tools node sets from the run's own history. That keeps a run
reproducible and safe to resume from a checkpoint on another replica.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from langchain_core.tools import BaseTool, tool


@dataclass(frozen=True)
class ToolContext:
    run_call_index: int = 0  # tool calls executed earlier in this run
    prior_same_calls: int = 0  # earlier calls with the same tool and arguments


TOOL_CONTEXT: ContextVar[ToolContext | None] = ContextVar("loopguard_tool_ctx", default=None)
_DEFAULT_CTX = ToolContext()


def current_context() -> ToolContext:
    return TOOL_CONTEXT.get() or _DEFAULT_CTX


_BASE_TIME = datetime(2026, 11, 18, 10, 0, 0, tzinfo=UTC)
_NS = uuid.UUID("6f1c2d6e-8a43-4f7a-9d0b-1c5a6e3b2f10")


def _now() -> str:
    idx = current_context().run_call_index
    return (_BASE_TIME + timedelta(seconds=7 * idx)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_id() -> str:
    return str(uuid.uuid5(_NS, f"req-{current_context().run_call_index}"))


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

ORDERS: dict[str, dict[str, object]] = {
    "1001": {
        "customer": "Asha Traders",
        "status": "shipped",
        "items": [("USB-C cable", 3, 299.0), ("Laptop stand", 1, 1499.0)],
    },
    "1002": {
        "customer": "Blue Lotus Cafe",
        "status": "delivered",
        "items": [("Espresso beans 1kg", 2, 1150.0)],
    },
    "1003": {
        "customer": "Kaveri Textiles",
        "status": "processing",
        "items": [("Cotton roll", 10, 640.0)],
    },
    "1004": {
        "customer": "Nexa Labs",
        "status": "delivered",
        "items": [("Monitor 27in", 1, 18999.0)],
    },
    "1005": {
        "customer": "Om Stationers",
        "status": "shipped",
        "items": [("Notebook A5", 25, 45.0), ("Gel pen pack", 10, 120.0)],
    },
    "1006": {
        "customer": "Pixel Studio",
        "status": "delivered",
        "items": [("SD card 128GB", 2, 1299.0)],
    },
    "1007": {
        "customer": "Riverstone Foods",
        "status": "processing",
        "items": [("Steel container", 4, 875.0)],
    },
    "1008": {
        "customer": "Sahyadri Hikes",
        "status": "shipped",
        "items": [("Rain jacket", 6, 2199.0)],
    },
    "1009": {
        "customer": "Tara Pharma",
        "status": "delivered",
        "items": [("Thermometer", 3, 349.0)],
    },
    "1010": {"customer": "Urban Roots", "status": "shipped", "items": [("Planter pot", 12, 260.0)]},
    "1011": {
        "customer": "Vega Motors",
        "status": "processing",
        "items": [("Tyre inflator", 1, 2899.0)],
    },
    "1012": {"customer": "Zen Yoga", "status": "delivered", "items": [("Yoga mat", 8, 799.0)]},
}


def order_subtotal(order_id: str) -> float:
    items = ORDERS[order_id]["items"]
    return float(sum(q * p for _, q, p in items))  # type: ignore[attr-defined]


WEATHER = {
    "pune": "27 C, clear sky, humidity 40%",
    "mumbai": "31 C, humid haze, humidity 78%",
    "bengaluru": "24 C, light rain, humidity 85%",
    "delhi": "19 C, smog advisory, humidity 55%",
    "hyderabad": "28 C, partly cloudy, humidity 50%",
}
WEATHER_OUTAGE = {"chennai", "kolkata", "jaipur", "ahmedabad", "kochi", "surat", "nagpur"}

# job-10xx finish: 1001-1003 on the third poll, 1004+ immediately. job-20xx never finish.
JOB_POLLS_TO_FINISH = {"job-1001": 3, "job-1002": 3, "job-1003": 3}

KB = {
    "kb-refund": (
        "Refund policy",
        "Customers can request a refund within 30 days of delivery. Refunds go back to the "
        "original payment method within 7 business days.",
    ),
    "kb-leave": (
        "Annual leave policy",
        "Full-time employees receive 24 days of annual leave per year and 12 days of sick "
        "leave. Up to 10 unused annual leave days carry over.",
    ),
    "kb-sla": (
        "Support SLA",
        "P1 incidents get a first response time of 15 minutes and a resolution target of "
        "4 hours. P2 incidents get a first response time of 2 hours.",
    ),
    "kb-travel": (
        "Travel reimbursement",
        "Economy airfare and hotels up to INR 6000 per night are reimbursed when receipts "
        "are submitted within 15 days of the trip.",
    ),
    "kb-security": (
        "Security incident response",
        "Report a suspected breach to the security team within 1 hour. The on-call "
        "engineer opens a P1 incident and starts the runbook.",
    ),
    "kb-onboarding": (
        "Onboarding checklist",
        "New hires receive a laptop on their first morning and finish compliance training "
        "during their first fortnight.",
    ),
}
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "is",
    "what",
    "how",
    "and",
    "or",
    "my",
    "our",
    "do",
    "does",
    "are",
    "with",
    "at",
    "by",
    "it",
    "be",
    "can",
    "we",
    "i",
}
KB_MIN_SCORE = 0.6

ORG = {
    "asha patil": ("Rohan Mehta", "Engineering"),
    "dev kulkarni": ("Sana Qureshi", "Finance"),
    "farah khan": ("Rohan Mehta", "Engineering"),
}
ORG_ORPHANS = {"vikram singh", "neha joshi", "arjun rao", "priya nair", "karan shah", "meera iyer"}

TICKET_PAGES = {"alpha": 3, "beta": 2, "gamma": 1}
TICKETS_PER_PAGE = 5


def _doc_body(doc_id: str) -> str:
    """A long, deterministic report body (about 8,000 characters)."""
    para = (
        f"[{doc_id}] Section {{n}}. Revenue, operating cost and headcount were reviewed "
        "against plan. The committee noted steady growth in the western region, higher "
        "logistics cost during the festive season, and a hiring freeze in support. Action "
        "items were assigned to finance, operations and engineering with review dates. "
    )
    return "".join(para.format(n=i + 1) for i in range(24))


DOCUMENTS = {f"board-q{i}": f"Board report Q{i}" for i in range(1, 5)}
DOCUMENTS.update({f"audit-2025-{i}": f"Internal audit 2025 part {i}" for i in range(1, 5)})


# ---------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id. Returns customer, status, line items and subtotal in INR."""
    oid = order_id.strip().lstrip("#")
    order = ORDERS.get(oid)
    if order is None:
        return f"ERROR 404: order {oid} not found."
    items = [{"item": n, "qty": q, "unit_price_inr": p} for n, q, p in order["items"]]  # type: ignore[attr-defined]
    return json.dumps(
        {
            "order_id": oid,
            "customer": order["customer"],
            "status": order["status"],
            "items": items,
            "subtotal_inr": order_subtotal(oid),
        }
    )


_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_safe_eval(node.right)) > 10:
            raise ValueError("exponent too large")
        return float(_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return float(_OPS[type(node.op)](_safe_eval(node.operand)))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '2396 * 1.18'. Supports + - * / % ** and ()."""
    if len(expression) > 200:
        return "ERROR: expression too long."
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"ERROR: {exc}"
    return f"{round(value, 2)}"


@tool
def fetch_weather(city: str) -> str:
    """Get the current weather for a delivery city in India."""
    key = city.strip().lower()
    if key in WEATHER_OUTAGE:
        return (
            f"ERROR 503: weather upstream unavailable for {city.strip()} "
            f"(request_id={_request_id()}, at {_now()})"
        )
    if key in WEATHER:
        return f"{city.strip()}: {WEATHER[key]} (observed {_now()})"
    return f"ERROR 404: no weather station for {city.strip()}."


@tool
def check_job_status(job_id: str) -> str:
    """Check the status of a batch job such as a nightly export. Returns RUNNING or DONE."""
    jid = job_id.strip().lower()
    poll = current_context().prior_same_calls + 1
    if not re.fullmatch(r"job-\d{4}", jid):
        return f"ERROR 400: invalid job id {job_id!r}."
    finishes_at = JOB_POLLS_TO_FINISH.get(jid, 1 if jid.startswith("job-10") else None)
    if finishes_at is not None and poll >= finishes_at:
        return f"{jid}: status=DONE result=s3://reports/{jid}.csv checked_at={_now()}"
    return f"{jid}: status=RUNNING progress=unknown checked_at={_now()}"


def _kb_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2 and w not in _STOP}


@tool
def search_kb(query: str) -> str:
    """Search the internal policy knowledge base (refunds, leave, SLA, travel, security)."""
    terms = _kb_terms(query)
    hits = []
    for doc_id, (title, body) in KB.items():
        doc_terms = _kb_terms(f"{title} {body}")
        score = len(terms & doc_terms) / max(1, len(terms))
        if score >= KB_MIN_SCORE:
            hits.append((score, doc_id, title, body))
    if not hits:
        return f'No results found for "{query}". Try a different query.'
    hits.sort(reverse=True)
    lines = [f'Found {len(hits[:2])} result(s) for "{query}":']
    lines += [f"- [{d}] {t}: {b}" for _, d, t, b in hits[:2]]
    return "\n".join(lines)


@tool
def get_manager(employee: str) -> str:
    """Find an employee's manager from the org chart."""
    key = employee.strip().lower()
    ref = 4100 + current_context().run_call_index
    if key in ORG:
        manager, dept = ORG[key]
        return f"Manager of {employee.strip()}: {manager} ({dept})."
    if key in ORG_ORPHANS:
        return (
            f"No manager on file for {employee.strip()} in the org chart. The org chart "
            f"defers to the HR record: call get_hr_record. (lookup ref {ref})"
        )
    return f"ERROR 404: no employee named {employee.strip()}."


@tool
def get_hr_record(employee: str) -> str:
    """Read an employee's HR record."""
    key = employee.strip().lower()
    ref = 4100 + current_context().run_call_index
    if key in ORG:
        manager, dept = ORG[key]
        return f"HR record for {employee.strip()}: department {dept}, manager {manager}."
    if key in ORG_ORPHANS:
        return (
            f"HR record for {employee.strip()} is incomplete: the manager field says 'see org "
            f"chart'. Call get_manager. (lookup ref {ref})"
        )
    return f"ERROR 404: no HR record for {employee.strip()}."


@tool
def list_tickets(project: str, page: int = 1) -> str:
    """List open support tickets for a project, one page at a time."""
    proj = project.strip().lower()
    page = max(1, int(page))
    if proj.startswith("legacy-"):
        pages: int | None = None  # broken pagination: there is always another page
    elif proj in TICKET_PAGES:
        pages = TICKET_PAGES[proj]
    else:
        return f"ERROR 404: unknown project {project!r}."
    if pages is not None and page > pages:
        return json.dumps({"project": proj, "page": page, "tickets": [], "next_page": None})
    start = (page - 1) * TICKETS_PER_PAGE
    tickets = [f"{proj.upper()}-{start + i + 1}" for i in range(TICKETS_PER_PAGE)]
    next_page = page + 1 if pages is None or page < pages else None
    return json.dumps({"project": proj, "page": page, "tickets": tickets, "next_page": next_page})


@tool
def read_document(doc_id: str) -> str:
    """Read a long internal report by id, e.g. board-q1 or audit-2025-1."""
    did = doc_id.strip().lower()
    if did not in DOCUMENTS:
        return f"ERROR 404: no document {doc_id!r}."
    return f"{DOCUMENTS[did]}\n\n{_doc_body(did)}"


ALL_TOOLS: list[BaseTool] = [
    lookup_order,
    calculator,
    fetch_weather,
    check_job_status,
    search_kb,
    get_manager,
    get_hr_record,
    list_tickets,
    read_document,
]


def default_tools() -> list[BaseTool]:
    return list(ALL_TOOLS)
