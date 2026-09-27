"""
Part 2 — The Case Triage Assistant.

Given a free-text case intake note, `TriageCaseAssistant.triage()`:

  1. Formats the prompt template with the case id/note and calls the
     Part 1 `LLMClient`.
  2. If the client itself is exhausted (both providers failed, per its
     documented fail-safe path), immediately produces a safe,
     `requires_human_review=True` fail-safe record — there is no raw
     text to repair.
  3. Otherwise, validates the raw response against the TriageRecord
     schema (`orca.triage.validator.parse_and_validate`). On failure, it
     re-prompts with a corrective instruction (`corrective_prompt`) up to
     `max_schema_retries` additional times before failing safe.
  4. Runs guardrails (`orca.triage.guardrails.run_guardrails`) on
     whatever record it ends up with — including the fail-safe record —
     so the non-negotiable Critical override always holds.
  5. Logs a full audit record for every call: input note id, raw model
     output, validated output, whether a schema retry happened, whether
     a guardrail intervened, provider used, cost, tokens, and timestamp.
"""

from __future__ import annotations

from typing import List, Optional

from orca.audit.logger import AuditLogger, new_audit_record
from orca.triage.guardrails import run_guardrails
from orca.triage.schemas import SchemaValidationError, TriageRecord
from orca.triage.validator import corrective_prompt, parse_and_validate

FAIL_SAFE_TIER = "High"
FAIL_SAFE_FLAG = "automation_failure"
FAIL_SAFE_STEP = (
    "Escalate to a human caseworker immediately: automated triage was "
    "unavailable or could not produce a valid assessment for this case."
)


def _fail_safe_record(case_id: str) -> TriageRecord:
    """
    The documented fail-safe path for Part 2. Deliberately tier "High"
    (never "Low", so it can't be silently deprioritized), always requires
    human review, confidence 0.0, and an explicit "automation_failure"
    flag — the caseworker sees an honest "the system couldn't triage
    this, look at it yourself," never a fabricated confident answer.
    """
    return TriageRecord(
        case_id=case_id,
        urgency_tier=FAIL_SAFE_TIER,
        recommended_next_steps=[FAIL_SAFE_STEP],
        flags=[FAIL_SAFE_FLAG],
        confidence=0.0,
        requires_human_review=True,
    )


class TriageCaseAssistant:
    def __init__(
        self,
        client,
        prompt_template: str,
        audit_logger: Optional[AuditLogger] = None,
        max_schema_retries: int = 1,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.audit_logger = audit_logger or AuditLogger(output_path=None)
        self.max_schema_retries = max_schema_retries

    def triage(self, case_id: str, case_note: str, feature: str = "triage") -> TriageRecord:
        base_prompt = self.prompt_template.format(case_id=case_id, case_note=case_note)
        current_prompt = base_prompt

        completion_results = []
        raw_outputs: List[str] = []
        schema_retried = False
        record: Optional[TriageRecord] = None

        total_schema_attempts = self.max_schema_retries + 1

        for attempt in range(total_schema_attempts):
            completion = self.client.complete(current_prompt, case_note, feature=feature)
            completion_results.append(completion)

            if completion.fail_safe:
                # Client-level exhaustion (both providers down): nothing
                # to parse or repair, go straight to fail-safe.
                break

            raw_outputs.append(completion.text)

            try:
                record = parse_and_validate(completion.text, case_id)
                break
            except SchemaValidationError as exc:
                record = None
                if attempt < total_schema_attempts - 1:
                    schema_retried = True
                    current_prompt = corrective_prompt(base_prompt, exc.errors)
                continue

        last_completion = completion_results[-1] if completion_results else None

        fail_safe = record is None
        if fail_safe:
            record = _fail_safe_record(case_id)

        final_record, guardrail_intervened, guardrail_reasons = run_guardrails(record)

        total_attempts = sum(c.attempts for c in completion_results)
        total_cost = sum(c.cost_usd for c in completion_results)
        total_tokens = sum(c.input_tokens + c.output_tokens for c in completion_results)

        audit_record = new_audit_record(
            case_id=case_id,
            raw_model_output=raw_outputs[-1] if raw_outputs else None,
            validated_output=final_record.to_dict(),
            provider_used=last_completion.provider_used if last_completion else None,
            attempts=total_attempts,
            schema_retried=schema_retried,
            guardrail_intervened=guardrail_intervened,
            guardrail_reasons=guardrail_reasons,
            fail_safe=fail_safe,
            cost_usd=total_cost,
            tokens=total_tokens,
        )
        self.audit_logger.log(audit_record)

        return final_record