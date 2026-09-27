"""
Part 3 — Offline eval harness pipeline.

`run_eval(cases_path, prompt_template, config, audit_output_path)` builds
a fresh provider pair + LLMClient + TriageCaseAssistant from `config`, runs
every case in the labeled dataset through the full triage pipeline
(schema validation/repair + guardrails included), and returns an
`EvalSummary` plus the full list of per-case `EvalCaseResult`s so callers
(e.g. `eval/run_eval.py`) can render per-case tables as well as the
aggregate metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from orca.audit.logger import AuditLogger
from orca.config import AppConfig
from orca.evaluation.metrics import compute_metrics
from orca.llm.client import LLMClient
from orca.llm.fake_llm import FakeLLMProvider
from orca.triage.assistant import TriageCaseAssistant


@dataclass
class EvalCaseResult:
    case_id: str
    gold_tier: str
    predicted_tier: str
    correct: bool
    provider_used: str
    used_fallback: bool
    cost_usd: float
    tokens: int


@dataclass
class EvalSummary:
    n_cases: int
    agreement: float
    critical_false_negative_rate: float
    avg_cost_per_case: float
    avg_tokens_per_case: float
    fallback_used_count: int
    fallback_used_rate: float
    total_cost: float


def _load_cases(cases_path) -> List[dict]:
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_pipeline(
    prompt_template: str, config: AppConfig, audit_output_path: Optional[str]
) -> TriageCaseAssistant:
    primary = FakeLLMProvider(
        config.primary.name,
        config.primary.failure_rate,
        config.primary.malformed_rate,
        seed=config.primary.seed,
    )
    fallback = FakeLLMProvider(
        config.fallback.name,
        config.fallback.failure_rate,
        config.fallback.malformed_rate,
        seed=config.fallback.seed,
    )
    client = LLMClient(
        primary,
        fallback,
        config.primary,
        config.fallback,
        max_retries=config.max_retries,
        backoff_base_seconds=config.backoff_base_seconds,
    )
    audit_logger = AuditLogger(output_path=audit_output_path)
    return TriageCaseAssistant(
        client=client,
        prompt_template=prompt_template,
        audit_logger=audit_logger,
        max_schema_retries=config.max_schema_retries,
    )


def run_eval(
    cases_path,
    prompt_template: str,
    config: AppConfig,
    audit_output_path: Optional[str] = None,
) -> Tuple[EvalSummary, List[EvalCaseResult]]:
    cases = _load_cases(cases_path)
    assistant = _build_pipeline(prompt_template, config, audit_output_path)

    raw_results: List[dict] = []
    case_results: List[EvalCaseResult] = []

    for case in cases:
        case_id = case["case_id"]
        note = case["case_note"]
        gold = case["gold_tier"]

        record = assistant.triage(case_id, note)
        audit = assistant.audit_logger.all_records()[-1]

        predicted = record.urgency_tier
        provider_used = audit.provider_used or "none (fail-safe)"
        used_fallback = bool(audit.provider_used and "fallback" in audit.provider_used.lower())

        raw_results.append(
            {
                "gold_tier": gold,
                "predicted_tier": predicted,
                "cost": audit.cost_usd,
                "tokens": audit.tokens,
                "used_fallback": used_fallback,
                "retried": audit.schema_retried,
                "guardrail_intervened": audit.guardrail_intervened,
            }
        )

        case_results.append(
            EvalCaseResult(
                case_id=case_id,
                gold_tier=gold,
                predicted_tier=predicted,
                correct=(predicted == gold),
                provider_used=provider_used,
                used_fallback=used_fallback,
                cost_usd=audit.cost_usd,
                tokens=audit.tokens,
            )
        )

    metrics = compute_metrics(raw_results)

    summary = EvalSummary(
        n_cases=metrics["n_cases"],
        agreement=metrics["tier_agreement"],
        critical_false_negative_rate=metrics["critical_false_negative_rate"],
        avg_cost_per_case=metrics["avg_cost_per_case_usd"],
        avg_tokens_per_case=metrics["avg_tokens_per_case"],
        fallback_used_count=metrics["fallback_invocations"],
        fallback_used_rate=metrics["fallback_rate"],
        total_cost=metrics["total_cost_usd"],
    )
    return summary, case_results