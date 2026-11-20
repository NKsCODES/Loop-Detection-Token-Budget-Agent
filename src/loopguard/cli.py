"""Command line: ``loopguard serve``, ``loopguard eval``, ``loopguard run``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from loopguard.config import Settings

app = typer.Typer(add_completion=False, help="LoopGuard agent: serve, evaluate, or run a task.")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, reload: bool = False) -> None:
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run(
        "loopguard.api.app:app_factory",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        proxy_headers=True,
    )


@app.command("eval")
def evaluate(
    out: Path = typer.Option(Path("reports"), help="Output directory."),
    provider: str = typer.Option("fake", help="fake | openai | anthropic"),
    model: str = typer.Option("", help="Model name for real providers."),
    sweep: bool = typer.Option(True, help="Also run the repeat-threshold sweep."),
    max_stuck_rate: float = typer.Option(
        1.0, help="Exit non-zero if LoopGuard's stuck rate is above this (CI gate)."
    ),
) -> None:
    """Run the 50-task stuck-rate benchmark."""
    from loopguard.agent.llm import build_chat_model
    from loopguard.eval.report import write_reports
    from loopguard.eval.runner import run_benchmark, threshold_variants

    settings = Settings(llm_provider=provider, llm_model=model)  # type: ignore[arg-type]
    chat = build_chat_model(settings)
    bench = run_benchmark(chat, pricing=settings.pricing)
    sweep_res = (
        run_benchmark(chat, variants=threshold_variants(), pricing=settings.pricing)
        if sweep
        else None
    )
    files = write_reports(bench, out, sweep_res)
    for name, data in bench["variants"].items():
        s = data["summary"]
        typer.echo(
            f"{name:16s} stuck {s['stuck']:2d}/{s['tasks']} ({s['stuck_rate']:.0%})  "
            f"passed {s['passed']:2d}  tokens {s['total_tokens']:>8,}  reviews {s['reviews']}"
        )
    typer.echo(f"reports written to {', '.join(str(p) for p in files.values())}")
    rate = bench["variants"]["loopguard"]["summary"]["stuck_rate"]
    if rate > max_stuck_rate:
        typer.echo(f"FAIL: stuck rate {rate:.0%} is above the gate {max_stuck_rate:.0%}", err=True)
        raise typer.Exit(1)


@app.command()
def run(task: str, auto_approve: bool = False) -> None:
    """Run one task in the terminal and answer reviews interactively."""
    from loopguard.agent.graph import ReviewDecision
    from loopguard.service import AgentService

    settings = Settings(database_url="sqlite://", checkpoint_url="memory")
    service = AgentService(settings)
    view = service.start_run(task)
    while view["status"] == "paused":
        p = view["pending"]
        typer.echo(f"\nPAUSED at step {p['step']}: [{p['kind']}/{p['detector']}] {p['message']}")
        for e in p.get("evidence", []):
            typer.echo(f"  step {e['step']}: {e['tool']}({e['args']}) -> {e['output'][:90]}")
        default = p["allowed_actions"][0]
        action = (
            default
            if auto_approve
            else typer.prompt(f"Action {p['allowed_actions']}", default=default)
        )
        view = service.resume_run(view["run_id"], ReviewDecision(action=action, reviewer="cli"))
    typer.echo(
        json.dumps(
            {
                k: view[k]
                for k in (
                    "status",
                    "stop_reason",
                    "final_answer",
                    "steps",
                    "total_tokens",
                    "cost_usd",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
