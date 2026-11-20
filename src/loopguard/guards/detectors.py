"""Loop detectors.

Each detector reads the tool-call history inside the current detection window and returns a
``LoopSignal`` or ``None``. They are pure functions: no I/O, no clocks, no randomness. That
matters because LangGraph re-runs a node when it resumes after an interrupt, so the guard must
reach the same verdict both times.

Detectors, in priority order:

* ``exact_repeat``  same tool, same args, same (normalised) output N times.
* ``cycle``         the input sequence ends in a repeating pattern, e.g. A B A B A B.
                    Catches ping-pong even when outputs carry a changing nonce.
* ``stagnation``    same tool keeps returning the same output for different args,
                    e.g. a search rephrased five times, "no results" every time.
* ``tool_overuse``  one tool called more than ``max_calls_per_tool`` times. The backstop for
                    loops whose inputs and outputs both change, such as runaway pagination.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from loopguard.config import GuardConfig
from loopguard.guards.fingerprint import ToolRecord


@dataclass(frozen=True)
class LoopSignal:
    detector: str
    tool: str
    count: int
    message: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evidence(records: Sequence[ToolRecord], limit: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "step": r.step,
            "tool": r.tool,
            "args": r.args_preview,
            "output": r.output_preview,
        }
        for r in list(records)[-limit:]
    ]


def detect_exact_repeat(records: Sequence[ToolRecord], cfg: GuardConfig) -> LoopSignal | None:
    counts = Counter(r.call_fp for r in records)
    # Check the newest record first so the message names the call that just tipped it over.
    for rec in reversed(records):
        n = counts[rec.call_fp]
        if n >= cfg.max_identical_calls:
            same = [r for r in records if r.call_fp == rec.call_fp]
            return LoopSignal(
                detector="exact_repeat",
                tool=rec.tool,
                count=n,
                message=(
                    f"`{rec.tool}` was called {n} times with the same arguments and returned "
                    "the same result each time."
                ),
                evidence=_evidence(same),
            )
    return None


def detect_cycle(records: Sequence[ToolRecord], cfg: GuardConfig) -> LoopSignal | None:
    seq = [r.args_fp for r in records]
    for period in range(2, cfg.cycle_max_period + 1):
        span = period * cfg.cycle_min_repeats
        if len(seq) < span:
            break
        tail = seq[-span:]
        pattern = tail[:period]
        if len(set(pattern)) < 2:
            continue  # all-same pattern is a plain repeat, handled elsewhere
        if all(tail[i] == pattern[i % period] for i in range(span)):
            tools = " -> ".join(r.tool for r in records[-span:][:period])
            return LoopSignal(
                detector="cycle",
                tool=records[-1].tool,
                count=cfg.cycle_min_repeats,
                message=(
                    f"The agent repeated the cycle [{tools}] {cfg.cycle_min_repeats} times "
                    "without making progress."
                ),
                evidence=_evidence(records[-span:], limit=span),
            )
    return None


def detect_stagnation(records: Sequence[ToolRecord], cfg: GuardConfig) -> LoopSignal | None:
    groups: dict[tuple[str, str], list[ToolRecord]] = defaultdict(list)
    for r in records:
        groups[(r.tool, r.output_fp)].append(r)
    for (tool, _), group in groups.items():
        distinct_inputs = {r.args_fp for r in group}
        if len(distinct_inputs) >= cfg.max_same_output:
            return LoopSignal(
                detector="stagnation",
                tool=tool,
                count=len(distinct_inputs),
                message=(
                    f"`{tool}` returned the same result for {len(distinct_inputs)} different "
                    "inputs. Rephrasing is not producing new information."
                ),
                evidence=_evidence(group),
            )
    return None


def detect_tool_overuse(records: Sequence[ToolRecord], cfg: GuardConfig) -> LoopSignal | None:
    counts = Counter(r.tool for r in records)
    tool, n = counts.most_common(1)[0] if counts else ("", 0)
    if n >= cfg.max_calls_per_tool:
        return LoopSignal(
            detector="tool_overuse",
            tool=tool,
            count=n,
            message=f"`{tool}` was called {n} times, over the per-tool limit of "
            f"{cfg.max_calls_per_tool}.",
            evidence=_evidence([r for r in records if r.tool == tool]),
        )
    return None


DETECTORS: tuple[Callable[[Sequence[ToolRecord], GuardConfig], LoopSignal | None], ...] = (
    detect_exact_repeat,
    detect_cycle,
    detect_stagnation,
    detect_tool_overuse,
)


def detect_loop(records: Sequence[ToolRecord], cfg: GuardConfig) -> LoopSignal | None:
    for detector in DETECTORS:
        signal = detector(records, cfg)
        if signal is not None:
            return signal
    return None
