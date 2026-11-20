# syntax=docker/dockerfile:1.7
# ---------- build stage: resolve and install dependencies with uv ----------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /src
RUN uv venv /opt/venv

# Dependencies first so this layer is cached when only source code changes
COPY pyproject.toml README.md ./
RUN mkdir -p src/loopguard && touch src/loopguard/__init__.py \
 && uv pip install ".[openai,anthropic,postgres]" \
 && uv pip uninstall loopguard

COPY src ./src
RUN uv pip install --no-deps ".[openai,anthropic,postgres]"

# Bake the tokenizer vocabulary into the image so the guard never needs network at runtime
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---------- runtime stage: small, non-root ----------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="loopguard-agent" \
      org.opencontainers.image.description="Tool-calling agent with loop detection, token budgets and human approval"

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    LOOPGUARD_ENV=prod

WORKDIR /app
RUN mkdir -p /app/data && chown app:app /app/data
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "loopguard.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
