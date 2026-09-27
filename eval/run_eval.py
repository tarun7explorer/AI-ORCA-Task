"""
Part 3 — ORCA triage eval harness.

Runnable as a single command:

    python eval/run_eval.py

Runs the full triage pipeline (Parts 1 + 2, including retry/fallback and
schema-repair/guardrails) over the 10 labeled case notes in
`data/cases.json`, once with `prompts/prompt_v1.txt` and once with
`prompts/prompt_v2.txt`, and prints a side-by-side regression comparison:
tier agreement, Critical false-negative rate (the failure mode that
matters most), average cost/tokens per case, and fallback-usage rate for
each version, plus the v1 -> v2 delta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orca.config import AppConfig
from orca.evaluation.evaluator import run_eval

console = Console()

TIER_COLORS = {"Low": "green", "Medium": "yellow", "High": "orange3", "Critical": "bold red"}


def render_tier(tier: str) -> str:
    color = TIER_COLORS.get(tier, "white")
    return f"[{color}]{tier}[/{color}]"


def render_summary(label: str, summary, results) -> None:
    table = Table(title=f"Per-case results — {label}", box=box.SIMPLE_HEAVY)
    table.add_column("Case ID")
    table.add_column("Gold")
    table.add_column("Predicted")
    table.add_column("Match")
    table.add_column("Provider")
    table.add_column("Cost ($)", justify="right")

    for r in results:
        match = "[green]\u2713[/green]" if r.correct else "[red]\u2717[/red]"
        table.add_row(
            r.case_id,
            render_tier(r.gold_tier),
            render_tier(r.predicted_tier),
            match,
            r.provider_used,
            f"{r.cost_usd:.6f}",
        )
    console.print(table)

    summary_table = Table(title=f"Summary — {label}", box=box.ROUNDED)
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Cases evaluated", str(summary.n_cases))
    summary_table.add_row("Tier agreement", f"{summary.agreement:.1%}")
    summary_table.add_row("Critical false-negative rate", f"{summary.critical_false_negative_rate:.1%}")
    summary_table.add_row("Avg cost / case", f"${summary.avg_cost_per_case:.6f}")
    summary_table.add_row("Avg tokens / case", f"{summary.avg_tokens_per_case:.1f}")
    summary_table.add_row(
        "Fallback used",
        f"{summary.fallback_used_count}/{summary.n_cases} ({summary.fallback_used_rate:.1%})",
    )
    summary_table.add_row("Total run cost", f"${summary.total_cost:.6f}")
    console.print(summary_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="ORCA triage eval harness")
    parser.add_argument("--cases", default=str(ROOT / "data" / "cases.json"))
    args = parser.parse_args()

    config = AppConfig.default()

    with console.status("[bold cyan]Running eval — prompt v1..."):
        v1_template = (ROOT / "prompts" / "prompt_v1.txt").read_text(encoding="utf-8")
        summary_v1, results_v1 = run_eval(
            args.cases,
            v1_template,
            config,
            audit_output_path=str(ROOT / "output" / "audit_log_v1.jsonl"),
        )

    with console.status("[bold cyan]Running eval — prompt v2..."):
        v2_template = (ROOT / "prompts" / "prompt_v2.txt").read_text(encoding="utf-8")
        summary_v2, results_v2 = run_eval(
            args.cases,
            v2_template,
            config,
            audit_output_path=str(ROOT / "output" / "audit_log_v2.jsonl"),
        )

    console.print(Panel.fit("[bold]ORCA Triage — Prompt Regression Eval[/bold]", border_style="cyan"))
    render_summary("prompt_v1", summary_v1, results_v1)
    render_summary("prompt_v2", summary_v2, results_v2)

    diff_table = Table(title="v1 vs v2 delta", box=box.MINIMAL_DOUBLE_HEAD)
    diff_table.add_column("Metric")
    diff_table.add_column("v1", justify="right")
    diff_table.add_column("v2", justify="right")
    diff_table.add_column("Delta", justify="right")

    def _delta(a: float, b: float, fmt: str):
        d = b - a
        sign = "+" if d >= 0 else ""
        return fmt.format(a), fmt.format(b), f"{sign}{fmt.format(d)}"

    diff_table.add_row("Tier agreement", *_delta(summary_v1.agreement, summary_v2.agreement, "{:.1%}"))
    diff_table.add_row(
        "Critical false-negative rate",
        *_delta(
            summary_v1.critical_false_negative_rate,
            summary_v2.critical_false_negative_rate,
            "{:.1%}",
        ),
    )
    diff_table.add_row(
        "Avg cost / case",
        *_delta(summary_v1.avg_cost_per_case, summary_v2.avg_cost_per_case, "${:.6f}"),
    )
    console.print(diff_table)

    if summary_v2.critical_false_negative_rate > summary_v1.critical_false_negative_rate:
        console.print(
            Panel(
                "[bold red]REGRESSION WARNING:[/bold red] v2 increases the Critical "
                "false-negative rate relative to v1. Review before promoting v2.",
                border_style="red",
            )
        )


if __name__ == "__main__":
    main()