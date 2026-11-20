"""Run registry (SQLAlchemy Core). SQLite in dev, PostgreSQL in prod, same code.

LangGraph's checkpointer owns the agent state. This table owns what the API lists and filters
on: status, task, totals, and the review currently waiting.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("task", Text, nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("stop_reason", String(64)),
    Column("final_answer", Text),
    Column("pending", JSON),
    Column("guard_config", JSON, nullable=False),
    Column("steps", Integer, nullable=False, default=0),
    Column("total_tokens", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("is_stuck", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def _now() -> datetime:
    return datetime.now(UTC)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        if url in ("sqlite://", "sqlite:///:memory:"):
            return create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        path = url.removeprefix("sqlite:///")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False})
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


SCHEMA_LOCK_KEY = 7_243_001  # arbitrary, constant across replicas


class RunStore:
    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)
        self._create_schema()

    def _create_schema(self) -> None:
        """Create tables. On PostgreSQL, serialise this across replicas.

        Two pods starting at once both run CREATE TABLE IF NOT EXISTS, and PostgreSQL can
        still fail one of them with a unique violation on its catalog. A session-level
        advisory lock makes the second pod wait for the first.
        """
        if self.engine.dialect.name != "postgresql":
            metadata.create_all(self.engine)
            return
        with self.engine.connect() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": SCHEMA_LOCK_KEY})
            try:
                metadata.create_all(conn)
                conn.commit()
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": SCHEMA_LOCK_KEY})
                conn.commit()

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def create(self, run_id: str, task: str, guard_config: dict[str, Any]) -> None:
        now = _now()
        with self.engine.begin() as conn:
            conn.execute(
                runs.insert().values(
                    run_id=run_id,
                    task=task,
                    status="running",
                    guard_config=guard_config,
                    steps=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    is_stuck=0,
                    created_at=now,
                    updated_at=now,
                )
            )

    def claim(self, run_id: str, from_status: str = "paused") -> bool:
        """Atomically move a run from ``from_status`` to running.

        This is a compare-and-set in the database, so two reviewers (or two replicas) cannot
        resume the same paused run twice.
        """
        with self.engine.begin() as conn:
            res = conn.execute(
                update(runs)
                .where(runs.c.run_id == run_id, runs.c.status == from_status)
                .values(status="running", updated_at=_now())
            )
            return bool(res.rowcount == 1)

    def update(self, run_id: str, **fields: Any) -> None:
        fields["updated_at"] = _now()
        with self.engine.begin() as conn:
            conn.execute(update(runs).where(runs.c.run_id == run_id).values(**fields))

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().first()
            return dict(row) if row else None

    def list(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = select(runs).order_by(runs.c.created_at.desc()).limit(limit)
        if status:
            q = q.where(runs.c.status == status)
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(q).mappings().all()]

    def count(self, status: str) -> int:
        with self.engine.connect() as conn:
            q = select(func.count()).select_from(runs).where(runs.c.status == status)
            return int(conn.execute(q).scalar_one())
