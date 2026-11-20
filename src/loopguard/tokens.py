"""Token counting and usage accounting.

Order of preference for every LLM call:

1. ``usage_metadata`` reported by the provider (exact, what you are billed for).
2. tiktoken ``cl100k_base`` estimate, if the vocabulary can be loaded.
3. A 4-characters-per-token heuristic, so the guard never runs blind.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from loopguard.config import ModelPricing
from loopguard.logs import get_logger

log = get_logger(__name__)

PER_MESSAGE_OVERHEAD = 4  # role and separators, matches OpenAI's chat accounting


def message_text(msg: BaseMessage) -> str:
    content: Any = msg.content
    if isinstance(content, list):  # multi-part content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        content = " ".join(parts)
    text = str(content or "")
    if isinstance(msg, AIMessage) and msg.tool_calls:
        text += json.dumps([{"name": c["name"], "args": c["args"]} for c in msg.tool_calls])
    return text


class TokenCounter:
    def __init__(self, encoding: str = "cl100k_base", use_tiktoken: bool = True) -> None:
        self._enc: Any = None
        self.backend = "heuristic"
        if use_tiktoken:
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding(encoding)
                self.backend = "tiktoken"
            except Exception as exc:  # offline or vocabulary download blocked
                log.warning("tiktoken_unavailable", error=type(exc).__name__)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return max(1, math.ceil(len(text) / 4))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        return sum(self.count_text(message_text(m)) + PER_MESSAGE_OVERHEAD for m in messages)


def usage_delta(
    ai: AIMessage,
    prompt: Sequence[BaseMessage],
    counter: TokenCounter,
    pricing: ModelPricing,
) -> dict[str, float]:
    """Return the token and cost delta for one LLM call."""
    meta = getattr(ai, "usage_metadata", None)
    if meta and meta.get("input_tokens") is not None:
        inp = int(meta.get("input_tokens", 0))
        out = int(meta.get("output_tokens", 0))
        estimated = 0
    else:
        inp = counter.count_messages(prompt)
        out = counter.count_messages([ai])
        estimated = 1
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "cost_usd": pricing.cost(inp, out),
        "llm_calls": 1,
        "estimated_calls": estimated,
    }


def merge_usage(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float]:
    """LangGraph reducer: sum usage counters across LLM calls."""
    a = a or {}
    b = b or {}
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
