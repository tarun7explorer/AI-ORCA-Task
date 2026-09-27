import pytest

from orca.triage.guardrails import (
    apply_critical_override,
    apply_disallowed_language_guardrail,
    run_guardrails,
)
from orca.triage.schemas import TriageRecord


def test_critical_tier_forces_human_review_even_if_model_said_false():
    record = TriageRecord(
        case_id="c1",
        urgency_tier="Critical",
        recommended_next_steps=["Escalate immediately"],
        flags=[],
        confidence=0.9,
        requires_human_review=False,
    )
    updated = apply_critical_override(record)
    assert updated.requires_human_review is True


def test_non_critical_tier_is_not_forced():
    record = TriageRecord(
        case_id="c2",
        urgency_tier="Low",
        recommended_next_steps=["Close case"],
        flags=[],
        confidence=0.9,
        requires_human_review=False,
    )
    updated = apply_critical_override(record)
    assert updated.requires_human_review is False


def test_critical_override_is_idempotent_when_already_true():
    record = TriageRecord(
        case_id="c3",
        urgency_tier="Critical",
        recommended_next_steps=["Escalate"],
        flags=[],
        confidence=0.9,
        requires_human_review=True,
    )
    updated = apply_critical_override(record)
    assert updated.requires_human_review is True


@pytest.mark.parametrize(
    "term", ["diagnose", "diagnosis", "prescribe", "psychiatric disorder", "medication dosage"]
)
def test_disallowed_terms_are_detected(term):
    record = TriageRecord(
        case_id="c4",
        urgency_tier="Medium",
        recommended_next_steps=[f"Caseworker should {term} the client"],
        flags=[],
        confidence=0.6,
        requires_human_review=False,
    )
    result = apply_disallowed_language_guardrail(record)
    assert result.triggered is True
    assert result.record.requires_human_review is True
    assert "disallowed_clinical_language" in result.record.flags


def test_clean_steps_do_not_trigger_guardrail():
    record = TriageRecord(
        case_id="c5",
        urgency_tier="Medium",
        recommended_next_steps=["Schedule caseworker contact within 24 hours"],
        flags=[],
        confidence=0.6,
        requires_human_review=False,
    )
    result = apply_disallowed_language_guardrail(record)
    assert result.triggered is False
    assert result.record is record


def test_run_guardrails_combines_language_and_critical_override():
    record = TriageRecord(
        case_id="c6",
        urgency_tier="Critical",
        recommended_next_steps=["Do not attempt to diagnose the client; escalate now"],
        flags=[],
        confidence=0.9,
        requires_human_review=False,
    )
    final, triggered, reasons = run_guardrails(record)
    assert final.requires_human_review is True
    assert triggered is True
    assert any("diagnose" in r for r in reasons)
    assert "disallowed_clinical_language" in final.flags


def test_disallowed_language_detection_is_case_insensitive():
    record = TriageRecord(
        case_id="c7",
        urgency_tier="Low",
        recommended_next_steps=["Do Not Prescribe anything without physician sign-off"],
        flags=[],
        confidence=0.5,
        requires_human_review=False,
    )
    result = apply_disallowed_language_guardrail(record)
    assert result.triggered is True