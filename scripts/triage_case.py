"""
Triage a single ORCA case note end-to-end (Parts 1 + 2 wired together).

Usage:
    # Direct positional string (auto-generates case-id, defaults to v1):
    python scripts/triage_case.py "Client described feeling hopeless and mentioned a specific plan to harm themselves."

    # Explicit named flags:
    python scripts/triage_case.py --case-id CASE-123 \
      --note "Client reported their partner has access to a weapon and recent threats were made." \
      --prompt-version v1
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orca.audit.logger import AuditLogger
from orca.config import AppConfig
from orca.llm.client import LLMClient
from orca.llm.fake_llm import FakeLLMProvider
from orca.triage.assistant import TriageCaseAssistant

console = Console()

TIER_COLORS = {"Low": "green", "Medium": "yellow", "High": "orange3", "Critical": "bold red"}


def render_tier(tier: str) -> str:
    color = TIER_COLORS.get(tier, "white")
    return f"[{color}]{tier}[/{color}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a single ORCA case note")
    # Allows passing the case note directly as a positional string
    parser.add_argument(
        "positional_note",
        nargs="?",
        default=None,
        help="Case note text passed directly without flags",
    )
    # Optional flags for explicit control
    parser.add_argument(
        "--case-id",
        dest="case_id",
        default=None,
        help="Unique identifier for the case (auto-generated if omitted)",
    )
    parser.add_argument(
        "--note",
        dest="flag_note",
        default=None,
        help="Case note text (alternative to positional argument)",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help="Prompt version to use (default: v1)",
    )
    args = parser.parse_args()

    # Resolve note text from either the positional string or the --note flag
    note_text = args.positional_note or args.flag_note
    if not note_text:
        parser.error("A case note is required. Provide it as a positional argument or via --note.")

    # Resolve or generate case_id
    case_id = args.case_id or f"manual-{uuid.uuid4().hex[:6]}"

    config = AppConfig.default()
    prompt_path = ROOT / "prompts" / f"prompt_{args.prompt_version}.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    primary = FakeLLMProvider(
        config.primary.name,
        config.primary.failure_rate,
        config.primary.malformed_rate,
        seed=config.primary.seed,
    )
    fallback = FakeLLMProvider(
        config.fallback.name,
        config.fallback.failure_rate,
        config.fallback.malformed_rate,
        seed=config.fallback.seed,
    )
    client = LLMClient(
        primary,
        fallback,
        config.primary,
        config.fallback,
        max_retries=config.max_retries,
        backoff_base_seconds=config.backoff_base_seconds,
    )

    audit_logger = AuditLogger(output_path=str(ROOT / "output" / "audit_log.jsonl"))
    assistant = TriageCaseAssistant(
        client=client,
        prompt_template=prompt_template,
        audit_logger=audit_logger,
        max_schema_retries=config.max_schema_retries,
    )

    with console.status(f"[bold cyan]Triaging case {case_id}..."):
        record = assistant.triage(case_id, note_text)

    audit = assistant.audit_logger.all_records()[-1]

    console.print(
        Panel.fit(
            f"[bold]Case {case_id}[/bold] — {render_tier(record.urgency_tier)}",
            border_style="cyan",
        )
    )

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Urgency tier", render_tier(record.urgency_tier))
    table.add_row(
        "Requires human review",
        "[bold red]YES[/bold red]" if record.requires_human_review else "no",
    )
    table.add_row("Confidence", f"{record.confidence:.2f}")
    table.add_row("Flags", ", ".join(record.flags) or "-")
    table.add_row("Next steps", "\n".join(f"- {s}" for s in record.recommended_next_steps))
    table.add_row("Provider used", audit.provider_used or "none (fail-safe)")
    table.add_row("Attempts", str(audit.attempts))
    table.add_row("Schema retried", str(audit.schema_retried))
    table.add_row("Guardrail intervened", str(audit.guardrail_intervened))
    table.add_row("Cost (USD)", f"${audit.cost_usd:.6f}")
    console.print(table)

    console.print(json.dumps(record.to_dict(), indent=2))


if __name__ == "__main__":
    main()