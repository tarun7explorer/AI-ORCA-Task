from orca.triage.schemas import TriageRecord


def test_end_to_end_pipeline_produces_valid_audit_and_record(clean_client, assistant_factory):
    assistant = assistant_factory(clean_client, max_schema_retries=1)
    record = assistant.triage(
        "case-critical-01",
        "Client reported their partner has access to a weapon and recent threats were made.",
    )

    assert isinstance(record, TriageRecord)
    assert record.urgency_tier == "Critical"
    assert record.requires_human_review is True  # enforced regardless of model output

    audit = assistant.audit_logger.all_records()[-1]
    assert audit.case_id == "case-critical-01"
    assert audit.provider_used == "orca-primary"
    assert audit.fail_safe is False
    assert audit.validated_output["urgency_tier"] == "Critical"
    assert audit.validated_output["requires_human_review"] is True
    assert audit.timestamp is not None


def test_end_to_end_pipeline_low_tier_case(clean_client, assistant_factory):
    assistant = assistant_factory(clean_client, max_schema_retries=1)
    record = assistant.triage(
        "case-low-01",
        "Routine follow-up call completed. Client doing well, no concerns raised.",
    )
    assert record.urgency_tier == "Low"
    assert record.requires_human_review is False


def test_end_to_end_pipeline_with_failover(always_fail_primary_client, assistant_factory):
    assistant = assistant_factory(always_fail_primary_client, max_schema_retries=1)
    record = assistant.triage("case-failover-01", "New intake, general info request.")
    audit = assistant.audit_logger.all_records()[-1]
    assert audit.provider_used == "orca-fallback"
    assert record is not None


def test_end_to_end_pipeline_fail_safe_when_all_exhausted(both_exhausted_client, assistant_factory):
    assistant = assistant_factory(both_exhausted_client, max_schema_retries=1)
    record = assistant.triage("case-deadletter-01", "Unclear situation, unable to reach client.")
    assert record.urgency_tier == "High"  # fail-safe tier
    assert record.requires_human_review is True
    assert "automation_failure" in record.flags

    audit = assistant.audit_logger.all_records()[-1]
    assert audit.fail_safe is True
    assert audit.raw_model_output is None


def test_audit_trail_records_every_call(clean_client, assistant_factory):
    assistant = assistant_factory(clean_client, max_schema_retries=1)
    for i in range(3):
        assistant.triage(f"case-multi-{i}", "Case file note: documentation update only.")
    assert len(assistant.audit_logger.all_records()) == 3


def test_disallowed_language_in_model_output_is_neutralized(monkeypatch, clean_client, assistant_factory):
    import orca.triage.assistant as assistant_module

    original_parse = assistant_module.parse_and_validate

    def _inject_disallowed(raw_text, case_id):
        record = original_parse(raw_text, case_id)
        record.recommended_next_steps.append("Do not diagnose without physician sign-off")
        return record

    monkeypatch.setattr(assistant_module, "parse_and_validate", _inject_disallowed)

    assistant = assistant_factory(clean_client, max_schema_retries=1)
    record = assistant.triage("case-guardrail-01", "Client doing well, no concerns raised.")

    assert record.requires_human_review is True
    assert "disallowed_clinical_language" in record.flags