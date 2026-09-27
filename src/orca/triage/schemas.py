"""
Part 2 — Structured record and validation-error types for the ORCA Case
Triage Assistant.

`TriageRecord` is the validated, caseworker-facing output shape:

    {
      "case_id": "<from input>",
      "urgency_tier": "Low | Medium | High | Critical",
      "recommended_next_steps": ["<string>", ...],
      "flags": ["<string>", ...],
      "confidence": 0.0-1.0,
      "requires_human_review": true | false
    }

It is intentionally a plain, mutable dataclass rather than a frozen model:
guardrails (Part 2) need to mutate `flags` and `requires_human_review` in
place after the record is first parsed from the model's raw output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

ALLOWED_TIERS = {"Low", "Medium", "High", "Critical"}


class SchemaValidationError(Exception):
    """
    Raised when a raw LLM response cannot be parsed into a valid
    TriageRecord.

    `errors` is a list of machine-readable error codes (e.g.
    "missing_field:confidence", "invalid_urgency_tier",
    "invalid_confidence", "json_decode_error: <detail>") describing every
    validation failure found, not just the first one. This list is also
    what gets fed into `orca.triage.validator.corrective_prompt` to build
    the re-prompt sent back to the model.
    """

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) if self.errors else "schema validation failed")


@dataclass
class TriageRecord:
    case_id: str
    urgency_tier: str
    recommended_next_steps: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_human_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)