"""Chat model factory. Provider SDKs are optional extras, imported only when selected."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from loopguard.config import Settings


def build_chat_model(settings: Settings, resolver: Any = None) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "fake":
        from loopguard.agent.fake_llm import ScriptedChatModel
        from loopguard.eval.tasks import default_resolver
        from loopguard.tokens import TokenCounter

        return ScriptedChatModel(
            resolver=resolver or default_resolver(), counter=TokenCounter(use_tiktoken=False)
        )

    if not settings.llm_model:
        raise ValueError("LOOPGUARD_LLM_MODEL is required for real providers.")

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'loopguard[openai]'") from exc
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            base_url=settings.llm_base_url,
            max_retries=2,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'loopguard[anthropic]'") from exc
        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=2,
        )

    raise ValueError(f"unknown provider {provider!r}")
