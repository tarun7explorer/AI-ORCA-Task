import pytest


def test_primary_exhaustion_triggers_fallback(always_fail_primary_client, prompt_template):
    result = always_fail_primary_client.complete(
        prompt_template.format(case_id="f1", case_note="unsafe housing conditions"),
        "unsafe housing conditions",
        feature="triage",
    )
    assert result.used_fallback is True
    assert result.provider_used == "orca-fallback"
    assert result.fail_safe is False


def test_fallback_usage_is_tracked_separately(always_fail_primary_client, prompt_template):
    always_fail_primary_client.complete(
        prompt_template.format(case_id="f2", case_note="documentation update"),
        "documentation update",
        feature="triage",
    )
    counts = always_fail_primary_client.usage.provider_call_counts("triage")
    assert counts.get("orca-fallback") == 1
    assert "orca-primary" not in counts


def test_fallback_cost_uses_fallback_rate_table(always_fail_primary_client, prompt_template):
    result = always_fail_primary_client.complete(
        prompt_template.format(case_id="f3", case_note="check-in missed"),
        "check-in missed",
        feature="triage",
    )
    expected_cost = (
        (result.input_tokens / 1_000_000) * 0.50
        + (result.output_tokens / 1_000_000) * 1.50
    )
    assert result.cost_usd == pytest.approx(expected_cost)


def test_both_providers_exhausted_returns_failsafe(both_exhausted_client, prompt_template):
    result = both_exhausted_client.complete(
        prompt_template.format(case_id="f4", case_note="anything at all"),
        "anything at all",
        feature="triage",
    )
    assert result.fail_safe is True
    assert result.text is None
    assert result.error is not None