"""FastAPI application.

Endpoints
---------
POST /v1/runs                       start a run (returns completed, paused, rejected or aborted)
GET  /v1/runs?status=paused         the review queue
GET  /v1/runs/{run_id}              full run detail, tool calls and guard events
POST /v1/runs/{run_id}/decision     approve, reject or extend_budget on a paused run
GET  /healthz  /readyz  /metrics    probes and Prometheus
GET  /                              reviewer console
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from loopguard import __version__
from loopguard.agent.graph import ReviewDecision
from loopguard.api.schemas import RunCreate, RunSummary, RunView
from loopguard.config import Settings, get_settings
from loopguard.logs import configure_logging, get_logger
from loopguard.metrics import HTTP_LATENCY, REGISTRY
from loopguard.service import AgentService, RunConflict, RunNotFound

log = get_logger(__name__)


def create_app(settings: Settings | None = None, service: AgentService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or AgentService(settings)
        log.info("startup", version=__version__, provider=settings.llm_provider, env=settings.env)
        yield
        log.info("shutdown")

    app = FastAPI(
        title="LoopGuard Agent API",
        version=__version__,
        description="Tool-calling agent with loop detection, token budgets and human approval.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        route = request.scope.get("route")
        HTTP_LATENCY.labels(
            request.method, getattr(route, "path", "unmatched"), str(response.status_code)
        ).observe(time.perf_counter() - start)
        response.headers["x-request-id"] = request_id
        return response

    def svc(request: Request) -> AgentService:
        return request.app.state.service  # type: ignore[no-any-return]

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (
            x_api_key and secrets.compare_digest(x_api_key, settings.api_key)
        ):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")

    @app.exception_handler(RunNotFound)
    async def not_found(_: Request, exc: RunNotFound) -> JSONResponse:
        return JSONResponse({"detail": f"run {exc} not found"}, status_code=404)

    @app.exception_handler(RunConflict)
    async def conflict(_: Request, exc: RunConflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    # ------------------------------------------------------------------ probes

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz(service: AgentService = Depends(svc)) -> dict[str, str]:
        try:
            service.store.ping()
        except Exception as exc:
            raise HTTPException(503, f"database unavailable: {type(exc).__name__}") from exc
        return {"status": "ready"}

    @app.get("/metrics", tags=["ops"])
    def metrics(service: AgentService = Depends(svc)) -> Response:
        service.refresh_gauges()
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ runs

    v1 = [Depends(require_api_key)]

    @app.post("/v1/runs", response_model=RunView, status_code=201, dependencies=v1, tags=["runs"])
    def create_run(body: RunCreate, service: AgentService = Depends(svc)) -> dict[str, Any]:
        return service.start_run(body.task, body.guard.as_dict() if body.guard else None)

    @app.get("/v1/runs", response_model=list[RunSummary], dependencies=v1, tags=["runs"])
    def list_runs(
        status_: str | None = Query(
            None,
            alias="status",
            pattern="^(running|paused|completed|rejected|aborted|stuck|failed)$",
        ),
        limit: int = Query(50, ge=1, le=500),
        service: AgentService = Depends(svc),
    ) -> list[dict[str, Any]]:
        return service.list_runs(status_, limit)

    @app.get("/v1/runs/{run_id}", response_model=RunView, dependencies=v1, tags=["runs"])
    def get_run(run_id: str, service: AgentService = Depends(svc)) -> dict[str, Any]:
        return service.get_run(run_id)

    @app.post(
        "/v1/runs/{run_id}/decision", response_model=RunView, dependencies=v1, tags=["review"]
    )
    def decide(
        run_id: str, decision: ReviewDecision, service: AgentService = Depends(svc)
    ) -> dict[str, Any]:
        return service.resume_run(run_id, decision)

    @app.get("/v1/examples", dependencies=v1, tags=["runs"])
    def examples() -> list[dict[str, str]]:
        """A few benchmark prompts, handy for demos with the offline model."""
        from loopguard.eval.tasks import benchmark_tasks

        picks = (
            "gst-1001",
            "trap-503-chennai",
            "trap-poll-job-2001",
            "trap-search-pet",
            "trap-pingpong-vikram",
            "trap-pages-legacy-crm",
            "docs-board",
            "poll-ok-job-1001",
        )
        by_id = {t.id: t for t in benchmark_tasks()}
        return [{"id": i, "category": by_id[i].category, "prompt": by_id[i].prompt} for i in picks]

    # ------------------------------------------------------------------ console

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console() -> str:
        return files("loopguard.api").joinpath("static/console.html").read_text("utf-8")

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory loopguard.api.app:app_factory``."""
    return create_app()
