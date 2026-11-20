"""Runtime configuration.

Two layers:

* ``GuardConfig`` holds the guardrail thresholds. It is a plain Pydantic model so it can be
  overridden per run (API request body) and stored with the run.
* ``Settings`` is the process configuration, read from environment variables with the
  ``LOOPGUARD_`` prefix (12-factor style), e.g. ``LOOPGUARD_LLM_PROVIDER=openai``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Action = Literal["pause", "abort"]


class GuardConfig(BaseModel):
    """Thresholds for the loop detector, step limit and token budget."""

    enabled: bool = True
    detect_loops: bool = True
    enforce_budget: bool = True

    # Step limits
    max_steps: int = Field(20, ge=1, description="Hard cap on tool-calling rounds per run.")
    max_calls_per_tool: int = Field(
        8, ge=1, description="Cap on calls to any single tool inside the detection window."
    )

    # Loop detection
    max_identical_calls: int = Field(
        3, ge=2, description="Same tool + same args + same output this many times = loop."
    )
    max_same_output: int = Field(
        3, ge=2, description="Same tool returns the same output for this many different args."
    )
    cycle_min_repeats: int = Field(3, ge=2, description="Repeats of an A-B-A-B pattern = loop.")
    cycle_max_period: int = Field(4, ge=2, description="Longest cycle period searched.")

    # Token and cost budget
    token_budget: int = Field(20_000, ge=100, description="Hard token cap per run.")
    cost_budget_usd: float | None = Field(0.25, ge=0, description="Hard cost cap per run.")
    soft_budget_ratio: float = Field(0.8, gt=0, lt=1, description="Warn the model at this ratio.")
    budget_extension_tokens: int = Field(
        10_000, ge=0, description="Tokens added when a reviewer approves a budget pause."
    )

    # What happens when a rule fires
    on_loop: Action = "pause"
    on_budget: Action = "pause"
    on_step_limit: Action = "abort"

    def recursion_limit(self) -> int:
        """LangGraph superstep cap used as a last line of defence.

        Each tool round is three supersteps (agent, tools, guard). The margin keeps the
        framework cap from firing before our own step limit does.
        """
        return self.max_steps * 3 + 10


class ModelPricing(BaseModel):
    """USD price per one million tokens. Set these to your model's current list price."""

    input_per_million: float = 3.0
    output_per_million: float = 15.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOOPGUARD_", env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # LLM
    llm_provider: Literal["fake", "openai", "anthropic"] = "fake"
    llm_model: str = ""
    llm_base_url: str | None = None  # any OpenAI-compatible endpoint (vLLM, Ollama, Groq)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    pricing: ModelPricing = ModelPricing()

    # Persistence: sqlite path for dev, postgresql:// URL for prod
    database_url: str = "sqlite:///./data/loopguard.db"
    checkpoint_url: str = "sqlite:///./data/checkpoints.db"

    # API
    api_key: str | None = None  # when set, every /v1 call needs header X-API-Key
    cors_origins: list[str] = []

    # Tools
    tool_timeout_s: float = 20.0
    max_tool_output_chars: int = 8_000

    guard: GuardConfig = GuardConfig()


@lru_cache
def get_settings() -> Settings:
    return Settings()
