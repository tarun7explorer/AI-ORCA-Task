import json

import pytest

from orca.triage.schemas import SchemaValidationError
from orca.triage.validator import corrective_prompt, parse_and_validate


def test_valid_payload_parses_successfully():
    payload = json.dumps(
        {
            "urgency_tier": "High",
            "recommended_next_steps": ["Verify current safety and location"],
            "flags": ["needs_review"],
            "confidence": 0.8,
        }
    )
    record = parse_and_validate(payload, case_id="case-99")
    assert record.case_id == "case-99"
    assert record.urgency_tier == "High"
    assert record.confidence == 0.8
    assert record.requires_human_review is False


def test_malformed_non_json_raises():
    raw = "Sure, here's the assessment: tier high, steps: contact caseworker"
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate(raw, case_id="case-01")
    assert any("json_decode_error" in e for e in exc_info.value.errors)


def test_missing_required_field_raises():
    payload = json.dumps(
        {
            "urgency_tier": "Low",
            "recommended_next_steps": ["Close case"],
            "flags": [],
        }
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate(payload, case_id="case-02")
    assert any("missing_field:confidence" in e for e in exc_info.value.errors)


def test_invalid_urgency_tier_raises():
    payload = json.dumps(
        {
            "urgency_tier": "Severe",
            "recommended_next_steps": ["Escalate"],
            "flags": [],
            "confidence": 0.5,
        }
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate(payload, case_id="case-03")
    assert any("invalid_urgency_tier" in e for e in exc_info.value.errors)


def test_invalid_confidence_range_raises():
    payload = json.dumps(
        {
            "urgency_tier": "Medium",
            "recommended_next_steps": ["Follow up"],
            "flags": [],
            "confidence": 1.5,
        }
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        parse_and_validate(payload, case_id="case-04")
    assert any("invalid_confidence" in e for e in exc_info.value.errors)


def test_non_object_json_raises():
    with pytest.raises(SchemaValidationError):
        parse_and_validate(json.dumps(["not", "an", "object"]), case_id="case-05")


def test_corrective_prompt_includes_original_and_errors():
    base = "Base prompt text"
    prompt = corrective_prompt(base, ["missing_field:confidence"])
    assert "Base prompt text" in prompt
    assert "CORRECTION REQUIRED" in prompt
    assert "missing_field:confidence" in prompt


def test_assistant_reprompts_once_then_fails_safe(always_malformed_client, assistant_factory):
    assistant = assistant_factory(always_malformed_client, max_schema_retries=1)
    record = assistant.triage("case-fail", "Client requesting general information about services.")
    assert record.requires_human_review is True
    assert "automation_failure" in record.flags

    audit = assistant.audit_logger.all_records()[-1]
    assert audit.schema_retried is True
    assert audit.fail_safe is True