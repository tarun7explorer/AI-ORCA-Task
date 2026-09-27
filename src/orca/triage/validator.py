"""
Part 2 — Schema validation and corrective re-prompting.

`parse_and_validate` turns a raw LLM completion string into a validated
`TriageRecord`, raising `SchemaValidationError` (with a list of specific,
machine-readable error codes) if the text is not valid JSON, is not a JSON
object, is missing a required field, has an out-of-vocabulary
`urgency_tier`, or has a `confidence` outside [0.0, 1.0].

`corrective_prompt` builds the re-prompt sent back to the model after a
validation failure, embedding the original prompt plus the specific
errors found so the model has a concrete, actionable correction to make
(rather than a generic "try again").
"""

from __future__ import annotations

import json
from typing import List

from orca.triage.schemas import ALLOWED_TIERS, SchemaValidationError, TriageRecord

REQUIRED_FIELDS = ["urgency_tier", "recommended_next_steps", "flags", "confidence"]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_and_validate(raw_text: str, case_id: str) -> TriageRecord:
    """
    Parse and validate a raw LLM completion against the TriageRecord
    schema.

    Raises
    ------
    SchemaValidationError
        If the text is not valid JSON, not a JSON object, missing a
        required field, has an invalid urgency_tier, or has a
        confidence outside [0.0, 1.0]. `exc.errors` holds every
        machine-readable error code found.
    """
    cleaned = _strip_code_fences(raw_text)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError([f"json_decode_error: {exc}"]) from exc

    if not isinstance(payload, dict):
        raise SchemaValidationError(["invalid_payload_type: expected a JSON object"])

    missing = [f"missing_field:{name}" for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise SchemaValidationError(missing)

    errors: List[str] = []

    urgency_tier = payload["urgency_tier"]
    if urgency_tier not in ALLOWED_TIERS:
        errors.append(
            f"invalid_urgency_tier: {urgency_tier!r} not in {sorted(ALLOWED_TIERS)}"
        )

    next_steps = payload["recommended_next_steps"]
    if not isinstance(next_steps, list) or not all(isinstance(s, str) for s in next_steps):
        errors.append("invalid_recommended_next_steps: expected a list of strings")

    flags = payload["flags"]
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        errors.append("invalid_flags: expected a list of strings")

    confidence = payload["confidence"]
    is_number = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    if not is_number or not (0.0 <= float(confidence) <= 1.0):
        errors.append(f"invalid_confidence: {confidence!r} not in [0.0, 1.0]")

    if errors:
        raise SchemaValidationError(errors)

    return TriageRecord(
        case_id=case_id,
        urgency_tier=urgency_tier,
        recommended_next_steps=list(next_steps),
        flags=list(flags),
        confidence=float(confidence),
        requires_human_review=bool(payload.get("requires_human_review", False)),
    )


CORRECTIVE_TEMPLATE = (
    "{original_prompt}\n\n"
    "--- CORRECTION REQUIRED ---\n"
    "Your previous response could not be validated against the required schema.\n"
    "Validation errors:\n{errors_block}\n\n"
    "Return ONLY a single valid JSON object with EXACTLY these keys and nothing else:\n"
    '  "urgency_tier": one of "Low", "Medium", "High", "Critical"\n'
    '  "recommended_next_steps": a non-empty list of strings\n'
    '  "flags": a list of strings (may be empty)\n'
    '  "confidence": a float between 0.0 and 1.0\n'
    "Do not include markdown code fences, prose, or any text outside the JSON object."
)


def corrective_prompt(original_prompt: str, errors: List[str]) -> str:
    """Build a corrective re-prompt embedding the original prompt and the
    specific validation errors found, per Part 2's re-prompt requirement."""
    errors_block = "\n".join(f"- {e}" for e in errors)
    return CORRECTIVE_TEMPLATE.format(original_prompt=original_prompt, errors_block=errors_block)