"""A scripted, loop-prone chat model for offline evaluation.

Why a simulated model: the benchmark has to run in CI on every commit, offline, at zero
cost, and give the same number every time. A live LLM gives none of that. So the benchmark
runs this model by default, and the same harness runs a real model with one flag
(``--provider openai`` or ``--provider anthropic``).

The scripted model behaves like a naive tool-calling agent:

* On healthy tasks it follows a correct plan.
* On a trap it does what weak agents do: retries the failed call, polls again, rephrases the
  search, follows the circular reference, fetches the next page.
* When the guard injects a loop nudge, it recovers with probability ``p_recover`` (seeded per
  task and nudge, so results are reproducible). Otherwise it ignores the nudge and keeps going.

Policies read only the message history, never hidden state, so a run can be paused,
checkpointed and resumed in another process and the model continues exactly where it was.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from loopguard.guards.policy import LOOP_NUDGE_PREFIX
from loopguard.tokens import TokenCounter

# ---------------------------------------------------------------------------------------------
# What the model "sees"
# ---------------------------------------------------------------------------------------------


@dataclass
class Observation:
    task: str
    outputs: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)
    loop_nudges: int = 0
    last_is_loop_nudge: bool = False

    @property
    def last_output(self) -> str:
        return self.outputs[-1][2] if self.outputs else ""

    def calls_to(self, tool: str) -> int:
        return sum(1 for name, _, _ in self.outputs if name == tool)


def observe(messages: Sequence[BaseMessage]) -> Observation:
    task = ""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    obs = Observation(task="")
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = str(msg.content)
            if text.startswith(LOOP_NUDGE_PREFIX):
                obs.loop_nudges += 1
            elif not task and not text.startswith("[loopguard"):
                task = text
        elif isinstance(msg, AIMessage):
            for c in msg.tool_calls:
                calls[c["id"] or ""] = (c["name"], c["args"])
        elif isinstance(msg, ToolMessage):
            name, args = calls.get(msg.tool_call_id, (msg.name or "", {}))
            obs.outputs.append((name, args, str(msg.content)))
    obs.task = task
    last = messages[-1] if messages else None
    obs.last_is_loop_nudge = isinstance(last, HumanMessage) and str(last.content).startswith(
        LOOP_NUDGE_PREFIX
    )
    return obs


@dataclass(frozen=True)
class Act:
    tool: str | None = None
    args: dict[str, Any] | None = None
    text: str = ""

    @staticmethod
    def call(tool: str, **args: Any) -> Act:
        return Act(tool=tool, args=args)

    @staticmethod
    def answer(text: str) -> Act:
        return Act(text=text)


# ---------------------------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------------------------


@dataclass
class Policy:
    task_id: str
    final: str
    graceful: str | None = None  # answer given when it recovers from a loop
    p_recover: float = 0.8

    def recovers(self, obs: Observation) -> bool:
        if self.graceful is None or not obs.last_is_loop_nudge:
            return False
        rng = random.Random(f"{self.task_id}:nudge:{obs.loop_nudges}")
        return rng.random() < self.p_recover

    def decide(self, obs: Observation) -> Act:
        if self.recovers(obs):
            return Act.answer(self.graceful or "")
        return self.step(obs)

    def step(self, obs: Observation) -> Act:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class PlanPolicy(Policy):
    """Healthy task: run a fixed plan of tool calls, then answer."""

    plan: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def step(self, obs: Observation) -> Act:
        i = len(obs.outputs)
        if i < len(self.plan):
            tool, args = self.plan[i]
            return Act.call(tool, **args)
        return Act.answer(self.final)


@dataclass
class RetryPolicy(Policy):
    """Call the same tool until the output contains ``done_marker``."""

    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    done_marker: str = ""

    def step(self, obs: Observation) -> Act:
        if obs.outputs and self.done_marker in obs.last_output:
            return Act.answer(self.final)
        return Act.call(self.tool, **self.args)


@dataclass
class RephrasePolicy(Policy):
    """Search, and rephrase the query every time nothing comes back."""

    queries: list[str] = field(default_factory=list)
    found_marker: str = "Found "

    def step(self, obs: Observation) -> Act:
        if obs.outputs and obs.last_output.startswith(self.found_marker):
            return Act.answer(self.final)
        i = obs.calls_to("search_kb")
        return Act.call("search_kb", query=self.queries[i % len(self.queries)])


@dataclass
class PingPongPolicy(Policy):
    """Follow the reference between the org chart and the HR record."""

    employee: str = ""
    resolved_marker: str = "Manager of"

    def step(self, obs: Observation) -> Act:
        if obs.outputs and self.resolved_marker in obs.last_output:
            return Act.answer(self.final)
        last_tool = obs.outputs[-1][0] if obs.outputs else "get_hr_record"
        nxt = "get_hr_record" if last_tool == "get_manager" else "get_manager"
        return Act.call(nxt, employee=self.employee)


@dataclass
class PaginatePolicy(Policy):
    """Fetch pages until ``next_page`` is null."""

    project: str = ""

    def step(self, obs: Observation) -> Act:
        if obs.outputs:
            try:
                if json.loads(obs.last_output).get("next_page") is None:
                    return Act.answer(self.final)
            except json.JSONDecodeError:
                pass
        return Act.call("list_tickets", project=self.project, page=obs.calls_to("list_tickets") + 1)


@dataclass
class EchoPolicy(Policy):
    def step(self, obs: Observation) -> Act:
        return Act.answer(self.final)


UNKNOWN_TASK = EchoPolicy(
    task_id="unknown",
    final=(
        "The fake provider only runs the scripted benchmark tasks. Send one of them, or set "
        "LOOPGUARD_LLM_PROVIDER to openai or anthropic to use a real model."
    ),
)

Resolver = Callable[[Observation], Policy]


# ---------------------------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------------------------


class ScriptedChatModel(BaseChatModel):
    """LangChain chat model driven by a policy per task. Reports usage like a provider."""

    resolver: Any  # Resolver; typed Any to keep pydantic happy with callables
    counter: Any = None
    schema_tokens: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-naive-agent"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        counter = self.counter or TokenCounter(use_tiktoken=False)
        schemas = json.dumps([convert_to_openai_tool(t) for t in tools])
        return self.model_copy(update={"schema_tokens": counter.count_text(schemas)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        counter = self.counter or TokenCounter(use_tiktoken=False)
        obs = observe(messages)
        policy: Policy = self.resolver(obs)
        act = policy.decide(obs)
        n_ai = sum(1 for m in messages if isinstance(m, AIMessage))
        if act.tool:
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": act.tool,
                        "args": act.args or {},
                        "id": f"call_{n_ai + 1}",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            ai = AIMessage(content=act.text)
        inp = counter.count_messages(messages) + self.schema_tokens
        out = counter.count_messages([ai])
        ai.usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
        return ChatResult(generations=[ChatGeneration(message=ai)])
