"""Write benchmark artifacts: JSON results, a Markdown report and two PNG charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABELS = {
    "baseline": "Baseline ReAct (recursion_limit=25)",
    "step_limit_only": "Step limit only (10 rounds)",
    "loopguard": "LoopGuard (full)",
}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def markdown(bench: dict[str, Any], sweep: dict[str, Any] | None = None) -> str:
    v = bench["variants"]
    lines = ["# Stuck-rate benchmark", ""]
    lines += [
        "50 tasks: 20 healthy, 30 loop traps (6 each: retry after 503, polling a job that never",
        "finishes, empty search, circular records, runaway pagination). Offline scripted model,",
        "seeded, so every run gives the same numbers. Reviewer decisions come from a fixed policy.",
        "",
        "| Variant | Stuck | Stuck rate | Passed | Tokens | Cost (USD) | Reviews "
        "| Loop false positives |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, data in v.items():
        s = data["summary"]
        lines.append(
            f"| {LABELS.get(name, name)} | {s['stuck']}/{s['tasks']} | {_pct(s['stuck_rate'])} | "
            f"{s['passed']}/{s['tasks']} | {s['total_tokens']:,} | {s['total_cost_usd']:.2f} | "
            f"{s['reviews']} | {s['loop_false_positives']} |"
        )
    if "baseline" in v and "loopguard" in v:
        b, g = v["baseline"]["summary"], v["loopguard"]["summary"]
        saved = 1 - g["total_tokens"] / b["total_tokens"]
        trap_saved = 1 - g["trap_tokens"] / b["trap_tokens"]
        lines += [
            "",
            f"LoopGuard cut total tokens by {_pct(saved)} and tokens spent on trap tasks by "
            f"{_pct(trap_saved)}. Median detection step on traps: {g['median_detection_step']}.",
            f"Traps resolved with an honest answer after one hint: {g['traps_passed']}/"
            f"{g['trap_tasks']}. Traps stopped by the reviewer: {g['traps_stopped_by_reviewer']}.",
        ]
    if "loopguard" in v:
        lines += [
            "",
            "## LoopGuard by task category",
            "",
            "| Category | Tasks | Stuck | Passed | Avg tokens | Avg steps | First detector |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in v["loopguard"]["by_category"]:
            det = ", ".join(f"{k} x{n}" for k, n in row["detectors"].items()) or "none"
            lines.append(
                f"| {row['category']} | {row['tasks']} | {row['stuck']} | {row['passed']} | "
                f"{row['avg_tokens']:,} | {row['avg_steps']} | {det} |"
            )
    if sweep:
        lines += [
            "",
            "## Repeat threshold sweep",
            "",
            "N = repeats before a loop is flagged (exact repeat, stagnation and cycle).",
            "",
            "| N | Stuck | Passed | Tokens | Loop false positives | Median detection step |",
            "|---|---|---|---|---|---|",
        ]
        for name, data in sweep["variants"].items():
            s = data["summary"]
            lines.append(
                f"| {name.removeprefix('loopguard_n')} | {s['stuck']} | {s['passed']} | "
                f"{s['total_tokens']:,} | {s['loop_false_positives']} | "
                f"{s['median_detection_step']} |"
            )
    return "\n".join(lines) + "\n"


def charts(bench: dict[str, Any], out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # charts are optional
        return []
    names = list(bench["variants"])
    labels = [LABELS.get(n, n).replace(" (", "\n(") for n in names]
    stuck = [bench["variants"][n]["summary"]["stuck"] for n in names]
    tokens = [bench["variants"][n]["summary"]["total_tokens"] / 1000 for n in names]
    grey, blue = "#9aa3b2", "#2457d6"
    colors = [blue if n == "loopguard" else grey for n in names]
    paths = []
    for values, title, ylabel, fname, fmt in (
        (stuck, "Stuck runs out of 50 tasks", "stuck runs", "stuck_rate.png", "{:.0f}"),
        (tokens, "Total tokens across 50 tasks", "thousand tokens", "tokens.png", "{:.0f}k"),
    ):
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
        bars = ax.bar(labels, values, color=colors, width=0.55)
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_reports(
    bench: dict[str, Any], out_dir: Path, sweep: dict[str, Any] | None = None
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {name: data["summary"] for name, data in bench["variants"].items()}
    files = {
        "results": out_dir / "benchmark_results.json",
        "summary": out_dir / "benchmark_summary.json",
        "markdown": out_dir / "BENCHMARK.md",
    }
    files["results"].write_text(json.dumps(bench, indent=2, default=str))
    payload: dict[str, Any] = {"variants": summary}
    if sweep:
        payload["threshold_sweep"] = {n: d["summary"] for n, d in sweep["variants"].items()}
    files["summary"].write_text(json.dumps(payload, indent=2, default=str))
    files["markdown"].write_text(markdown(bench, sweep))
    for p in charts(bench, out_dir):
        files[p.stem] = p
    return files
