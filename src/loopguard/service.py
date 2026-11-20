"""Application service: start, resume and inspect guarded agent runs."""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from loopguard.agent.graph import ReviewDecision, build_agent_graph
from loopguard.agent.llm import build_chat_model
from loopguard.agent.runner import PAUSED, RunOutcome, invoke
from loopguard.agent.tools import default_tools
from loopguard.config import GuardConfig, Settings
from loopguard.logs import get_logger
from loopguard.metrics import PENDING_REVIEWS, RUNS_STARTED, record_invocation
from loopguard.store import RunStore
from loopguard.tokens import TokenCounter

log = get_logger(__name__)


CHECKPOINT_LOCK_KEY = 7_243_002


class RunNotFound(LookupError):
    pass


class RunConflict(RuntimeError):
    pass


def make_checkpointer(url: str) -> BaseCheckpointSaver[Any]:
    if url == "memory":
        return InMemorySaver()
    if url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = url.removeprefix("sqlite:///")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        saver.setup()
        return saver
    if url.startswith("postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'loopguard[postgres]'") from exc
        pool = ConnectionPool(
            url, max_size=10, kwargs={"autocommit": True, "prepare_threshold": 0}, open=True
        )
        pg = PostgresSaver(pool)  # type: ignore[arg-type]
        # Run the checkpointer migrations under an advisory lock so replicas that start
        # together do not race on CREATE TABLE.
        with pool.connection() as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (CHECKPOINT_LOCK_KEY,))
            try:
                pg.setup()
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (CHECKPOINT_LOCK_KEY,))
        return pg
    raise ValueError(f"unsupported checkpoint url: {url}")


class AgentService:
    def __init__(
        self,
        settings: Settings,
        model: BaseChatModel | None = None,
        tools: Sequence[BaseTool] | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or RunStore(settings.database_url)
        self.checkpointer = checkpointer or make_checkpointer(settings.checkpoint_url)
        model = model or build_chat_model(settings)
        counter = TokenCounter(use_tiktoken=settings.llm_provider != "fake")
        self.graph = build_agent_graph(
            model,
            tools or default_tools(),
            guarded=True,
            checkpointer=self.checkpointer,
            guard_config=settings.guard,
            pricing=settings.pricing,
            counter=counter,
            tool_timeout_s=settings.tool_timeout_s,
            max_tool_output_chars=settings.max_tool_output_chars,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _config(run_id: str, guard: GuardConfig) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id, "guard": guard.model_dump()},
            "recursion_limit": guard.recursion_limit(),
        }

    def _values(self, run_id: str, guard: GuardConfig) -> dict[str, Any]:
        snap = self.graph.get_state(cast(RunnableConfig, self._config(run_id, guard)))
        return dict(snap.values or {})

    def _execute(self, run_id: str, payload: Any, guard: GuardConfig) -> RunOutcome:
        config = self._config(run_id, guard)
        before = self._values(run_id, guard)
        outcome = invoke(self.graph, payload, config)
        after = self._values(run_id, guard)
        record_invocation(
            before,
            after,
            outcome.status,
            outcome.pending,
            outcome.is_stuck,
            outcome.stop_reason,
            outcome.steps,
        )
        self.store.update(
            run_id,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            final_answer=outcome.final_answer,
            pending=outcome.pending,
            steps=outcome.steps,
            total_tokens=int(outcome.usage.get("total_tokens", 0)),
            cost_usd=float(outcome.usage.get("cost_usd", 0.0)),
            is_stuck=int(outcome.is_stuck),
        )
        log.info(
            "run_invocation",
            run_id=run_id,
            status=outcome.status,
            steps=outcome.steps,
            tokens=int(outcome.usage.get("total_tokens", 0)),
            pending=(outcome.pending or {}).get("detector"),
            error=outcome.error,
        )
        return outcome

    def _guard_for(self, row: dict[str, Any]) -> GuardConfig:
        return GuardConfig.model_validate(row["guard_config"])

    # ------------------------------------------------------------------ API

    def start_run(self, task: str, guard_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        guard = GuardConfig.model_validate(
            {**self.settings.guard.model_dump(), **(guard_overrides or {})}
        )
        run_id = str(uuid.uuid4())
        self.store.create(run_id, task, guard.model_dump())
        RUNS_STARTED.inc()
        self._execute(run_id, {"messages": [HumanMessage(content=task)]}, guard)
        return self.get_run(run_id)

    def resume_run(self, run_id: str, decision: ReviewDecision) -> dict[str, Any]:
        row = self.store.get(run_id)
        if row is None:
            raise RunNotFound(run_id)
        pending = row.get("pending") or {}
        allowed = pending.get("allowed_actions") or []
        # "approve" on a budget pause means extend, so it is accepted there too
        approve_means_extend = decision.action == "approve" and "extend_budget" in allowed
        if (
            row["status"] == PAUSED
            and allowed
            and decision.action not in allowed
            and not approve_means_extend
        ):
            raise RunConflict(f"action {decision.action!r} not allowed; use one of {allowed}")
        if not self.store.claim(run_id, from_status=PAUSED):
            raise RunConflict(f"run {run_id} is {row['status']}, not paused")
        self._execute(run_id, Command(resume=decision.model_dump()), self._guard_for(row))
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.store.get(run_id)
        if row is None:
            raise RunNotFound(run_id)
        values = self._values(run_id, self._guard_for(row))
        return {
            **{
                k: row[k]
                for k in (
                    "run_id",
                    "task",
                    "status",
                    "stop_reason",
                    "final_answer",
                    "pending",
                    "steps",
                    "total_tokens",
                    "cost_usd",
                )
            },
            "is_stuck": bool(row["is_stuck"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "usage": values.get("usage", {}),
            "guard_events": values.get("guard_events", []),
            "tool_calls": [
                {k: r[k] for k in ("step", "tool", "args_preview", "output_preview", "is_error")}
                for r in values.get("tool_records", [])
            ],
            "guard_config": row["guard_config"],
        }

    def list_runs(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                k: r[k]
                for k in (
                    "run_id",
                    "task",
                    "status",
                    "stop_reason",
                    "steps",
                    "total_tokens",
                    "cost_usd",
                    "created_at",
                    "updated_at",
                    "pending",
                )
            }
            for r in self.store.list(status, limit)
        ]

    def refresh_gauges(self) -> None:
        PENDING_REVIEWS.set(self.store.count(PAUSED))
