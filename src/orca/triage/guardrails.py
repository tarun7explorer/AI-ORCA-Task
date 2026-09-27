"""
Part 2 — Guardrails enforced in code, not merely requested in the prompt.

Two independent guardrails:

1. Non-negotiable Critical override (`apply_critical_override`): if
   `urgency_tier == "Critical"`, `requires_human_review` is forced to
   `True` regardless of what the model output said. This is enforced
   here, in code, and cannot be bypassed by the model.

2. Disallowed-language guardrail (`apply_disallowed_language_guardrail`):
   `recommended_next_steps` is scanned case-insensitively against the
   illustrative `DISALLOWED_TERMS` list from Appendix B. Enforcement
   strategy implemented: pass the content through unmodified but flag it
   (`disallowed_clinical_language`) and force `requires_human_review`
   to `True`. See docs/design_decisions.md, Q1, for the justification —
   stripping risks silently discarding a genuine safety signal, and
   reject-and-retry risks looping indefinitely without ever surfacing
   the concern to a human.

`run_guardrails` composes both, in order, and reports which ones fired
and why, for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from orca.triage.schemas import TriageRecord

DISALLOWED_TERMS = [
    "diagnose",
    "diagnosis",
    "prescribe",
    "psychiatric disorder",
    "medication dosage",
]

FLAG_DISALLOWED_LANGUAGE = "disallowed_clinical_language"
REASON_CRITICAL_OVERRIDE = "critical_tier_override"


@dataclass
class DisallowedLanguageOutcome:
    triggered: bool
    record: TriageRecord


def _find_disallowed_hits(steps: List[str]) -> List[str]:
    hits: List[str] = []
    for step in steps:
        step_lower = step.lower()
        for term in DISALLOWED_TERMS:
            if term in step_lower and term not in hits:
                hits.append(term)
    return hits


def apply_critical_override(record: TriageRecord) -> TriageRecord:
    """Non-negotiable: Critical tier always requires human review, enforced
    here in code regardless of what the model set requires_human_review to."""
    if record.urgency_tier == "Critical" and not record.requires_human_review:
        record.requires_human_review = True
    return record


def apply_disallowed_language_guardrail(record: TriageRecord) -> DisallowedLanguageOutcome:
    """Scan recommended_next_steps for disallowed clinical/diagnostic
    language (case-insensitive). On a hit: flag the record and force
    human review, but do not alter the underlying text (pass-through
    with mandatory review — see docs/design_decisions.md Q1)."""
    hits = _find_disallowed_hits(record.recommended_next_steps)
    if not hits:
        return DisallowedLanguageOutcome(triggered=False, record=record)

    if FLAG_DISALLOWED_LANGUAGE not in record.flags:
        record.flags.append(FLAG_DISALLOWED_LANGUAGE)
    record.requires_human_review = True
    return DisallowedLanguageOutcome(triggered=True, record=record)


def run_guardrails(record: TriageRecord) -> Tuple[TriageRecord, bool, List[str]]:
    """
    Run every guardrail against `record` in order, mutating and returning
    it, plus whether any guardrail intervened and the specific reasons
    (for the audit trail).
    """
    reasons: List[str] = []
    triggered = False

    if record.urgency_tier == "Critical" and not record.requires_human_review:
        triggered = True
        reasons.append(REASON_CRITICAL_OVERRIDE)
    record = apply_critical_override(record)

    language_outcome = apply_disallowed_language_guardrail(record)
    record = language_outcome.record
    if language_outcome.triggered:
        triggered = True
        for term in _find_disallowed_hits(record.recommended_next_steps):
            reasons.append(f"disallowed_term_detected:{term}")

    return record, triggered, reasons