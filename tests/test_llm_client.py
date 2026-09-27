import json

import pytest

from orca.config import ProviderConfig
from orca.llm.client import LLMClient
from orca.llm.fake_llm import FakeLLMProvider


def test_successful_call_records_usage_and_cost(clean_client, prompt_template):
    result = clean_client.complete(
        prompt_template.format(case_id="c1", case_note="Routine follow-up call completed."),
        "Routine follow-up call completed.",
        feature="triage",
    )
    assert result.fail_safe is False
    assert result.used_fallback is False
    assert result.provider_used == "orca-primary"
    assert result.attempts == 1
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.cost_usd > 0

    payload = json.loads(result.text)
    assert payload["urgency_tier"] in ("Low", "Medium", "High", "Critical")


def test_retry_on_transient_error_before_fallback(always_fail_primary_client, prompt_template):
    result = always_fail_primary_client.complete(
        prompt_template.format(case_id="c2", case_note="weapon threat"),
        "weapon threat",
        feature="triage",
    )
    assert result.used_fallback is True
    assert result.provider_used == "orca-fallback"
    # 2 primary attempts (max_retries=2) + 1 fallback attempt
    assert result.attempts == 3


def test_fail_safe_when_both_exhausted(both_exhausted_client, prompt_template):
    result = both_exhausted_client.complete(
        prompt_template.format(case_id="c3", case_note="anything"),
        "anything",
        feature="triage",
    )
    assert result.fail_safe is True
    assert result.text is None
    assert result.provider_used is None
    assert result.cost_usd == 0.0
    assert result.error is not None


def test_cumulative_usage_tracking_across_calls(clean_client, prompt_template):
    for i in range(3):
        clean_client.complete(
            prompt_template.format(case_id=f"c{i}", case_note="no concerns raised"),
            "no concerns raised",
            feature="triage",
        )
    assert clean_client.usage.total_tokens("triage") > 0
    assert clean_client.usage.total_cost("triage") > 0
    counts = clean_client.usage.provider_call_counts("triage")
    assert counts.get("orca-primary") == 3


def test_usage_attributable_to_feature(clean_client, prompt_template):
    clean_client.complete(prompt_template.format(case_id="c1", case_note="note"), "note", feature="triage")
    clean_client.complete(prompt_template.format(case_id="c1", case_note="note"), "note", feature="other_feature")

    triage_tokens = clean_client.usage.total_tokens("triage")
    other_tokens = clean_client.usage.total_tokens("other_feature")
    all_tokens = clean_client.usage.total_tokens()

    assert triage_tokens > 0
    assert other_tokens > 0
    assert all_tokens == triage_tokens + other_tokens


def test_cost_calculation_matches_rate_table():
    cfg = ProviderConfig("orca-primary", 0.0, 0.0, 3.00, 15.00, seed=1)
    fb_cfg = ProviderConfig("orca-fallback", 0.0, 0.0, 0.50, 1.50, seed=2)
    client = LLMClient(
        primary=FakeLLMProvider("orca-primary", 0.0, 0.0, seed=1),
        fallback=FakeLLMProvider("orca-fallback", 0.0, 0.0, seed=2),
        primary_cfg=cfg,
        fallback_cfg=fb_cfg,
    )
    result = client.complete("prompt text here", "note text", feature="triage")
    expected_cost = (
        (result.input_tokens / 1_000_000) * 3.00
        + (result.output_tokens / 1_000_000) * 15.00
    )
    assert result.cost_usd == pytest.approx(expected_cost)