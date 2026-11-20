from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from loopguard.agent.fake_llm import ScriptedChatModel
from loopguard.agent.graph import build_agent_graph
from loopguard.agent.tools import default_tools
from loopguard.config import GuardConfig, ModelPricing
from loopguard.eval.tasks import benchmark_tasks, default_resolver
from loopguard.tokens import TokenCounter


@pytest.fixture
def counter() -> TokenCounter:
    return TokenCounter(use_tiktoken=False)


@pytest.fixture
def model(counter: TokenCounter) -> ScriptedChatModel:
    return ScriptedChatModel(resolver=default_resolver(), counter=counter)


@pytest.fixture
def tasks():
    return {t.id: t for t in benchmark_tasks()}


@pytest.fixture
def make_graph(model, counter):
    def _make(guarded: bool = True, cfg: GuardConfig | None = None):
        return build_agent_graph(
            model,
            default_tools(),
            guarded=guarded,
            checkpointer=InMemorySaver(),
            guard_config=cfg or GuardConfig(),
            pricing=ModelPricing(),
            counter=counter,
        )

    return _make


def start_payload(prompt: str) -> dict:
    return {"messages": [HumanMessage(content=prompt)]}


def config_for(thread: str, cfg: GuardConfig | None = None, limit: int | None = None) -> dict:
    cfg = cfg or GuardConfig()
    return {
        "configurable": {"thread_id": thread, "guard": cfg.model_dump()},
        "recursion_limit": limit or cfg.recursion_limit(),
    }
