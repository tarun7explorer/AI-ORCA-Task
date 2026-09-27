"""
Part 3 — Aggregate eval metrics.

`compute_metrics` takes a list of per-case result dicts (each with at
least `gold_tier`, `predicted_tier`, `cost`, `tokens`, `used_fallback`,
`retried`, `guardrail_intervened`) and computes:

  - Overall tier agreement (plain accuracy).
  - Critical false-negative rate: among gold-Critical cases, the
    fraction where the prediction was NOT Critical. This is the failure
    mode that matters most — plain accuracy hides it, since a model that
    is very good at the common Low/Medium cases can still have a
    dangerous blind spot on the rare, highest-stakes Critical cases.
  - Average dollar cost and token count per case.
  - Fallback-invocation rate, schema-retry rate, and guardrail-
    intervention rate, for cost/reliability instrumentation (Part 4).
  - A full gold-tier x predicted-tier confusion matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List

TIER_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n_cases": 0,
            "tier_agreement": 0.0,
            "critical_false_negative_rate": 0.0,
            "critical_gold_count": 0,
            "critical_missed_count": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_case_usd": 0.0,
            "total_tokens": 0,
            "avg_tokens_per_case": 0.0,
            "fallback_invocations": 0,
            "fallback_rate": 0.0,
            "retry_rate": 0.0,
            "guardrail_intervention_rate": 0.0,
            "confusion": {},
        }

    agree = 0
    critical_gold_count = 0
    critical_missed = 0
    total_cost = 0.0
    total_tokens = 0
    fallback_invocations = 0
    retried_count = 0
    guardrail_count = 0
    confusion: Dict[str, Dict[str, int]] = {}

    for r in results:
        gold = r["gold_tier"]
        pred = r["predicted_tier"]

        confusion.setdefault(gold, {}).setdefault(pred, 0)
        confusion[gold][pred] += 1

        if gold == pred:
            agree += 1

        if gold == "Critical":
            critical_gold_count += 1
            if pred != "Critical":
                critical_missed += 1

        total_cost += r.get("cost", 0.0)
        total_tokens += r.get("tokens", 0)
        if r.get("used_fallback"):
            fallback_invocations += 1
        if r.get("retried"):
            retried_count += 1
        if r.get("guardrail_intervened"):
            guardrail_count += 1

    critical_fnr = (critical_missed / critical_gold_count) if critical_gold_count else 0.0

    return {
        "n_cases": n,
        "tier_agreement": agree / n,
        "critical_false_negative_rate": critical_fnr,
        "critical_gold_count": critical_gold_count,
        "critical_missed_count": critical_missed,
        "total_cost_usd": total_cost,
        "avg_cost_per_case_usd": total_cost / n,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": total_tokens / n,
        "fallback_invocations": fallback_invocations,
        "fallback_rate": fallback_invocations / n,
        "retry_rate": retried_count / n,
        "guardrail_intervention_rate": guardrail_count / n,
        "confusion": confusion,
    }