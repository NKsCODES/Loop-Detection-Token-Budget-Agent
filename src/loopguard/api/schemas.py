"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GuardOverrides(BaseModel):
    """Per-run overrides. Anything left out uses the server default."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(None, ge=1, le=200)
    max_calls_per_tool: int | None = Field(None, ge=1, le=200)
    max_identical_calls: int | None = Field(None, ge=2, le=50)
    max_same_output: int | None = Field(None, ge=2, le=50)
    cycle_min_repeats: int | None = Field(None, ge=2, le=50)
    token_budget: int | None = Field(None, ge=100, le=5_000_000)
    cost_budget_usd: float | None = Field(None, ge=0, le=100)
    on_loop: str | None = Field(None, pattern="^(pause|abort)$")
    on_budget: str | None = Field(None, pattern="^(pause|abort)$")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class RunCreate(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    guard: GuardOverrides | None = None


class PendingReview(BaseModel):
    kind: str
    detector: str | None = None
    message: str
    evidence: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    step: int
    usage: dict[str, Any] = {}
    allowed_actions: list[str]


class ToolCallView(BaseModel):
    step: int
    tool: str
    args_preview: str
    output_preview: str
    is_error: bool


class RunView(BaseModel):
    run_id: str
    task: str
    status: str
    stop_reason: str | None = None
    final_answer: str | None = None
    pending: PendingReview | None = None
    steps: int
    total_tokens: int
    cost_usd: float
    is_stuck: bool
    usage: dict[str, Any] = {}
    guard_events: list[dict[str, Any]] = []
    tool_calls: list[ToolCallView] = []
    guard_config: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


class RunSummary(BaseModel):
    run_id: str
    task: str
    status: str
    stop_reason: str | None = None
    steps: int
    total_tokens: int
    cost_usd: float
    pending: PendingReview | None = None
    created_at: datetime
    updated_at: datetime
