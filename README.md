# ORCA AI Layer

Triage assistant, guardrails, and eval harness for SRN's ORCA case-management platform.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No environment variables or API keys are required — Part 1's `LLMClient` wraps two
`FakeLLMProvider` instances (Appendix B) so everything runs offline and deterministically.

## Run a single case

```bash
python scripts/triage_case.py --case-id CASE-123 \
  --note "Client reported their partner has access to a weapon and recent threats were made." \
  --prompt-version v1
```

This prints the triaged record (tier, human-review flag, confidence, flags, next steps),
the audit metadata for that call (provider used, attempts, retries, guardrail interventions,
cost), and appends a JSON line to `output/audit_log.jsonl`.

## Run the eval harness (v1 vs v2 regression check)

```bash
python eval/run_eval.py
```

Runs the full triage pipeline over the 10 labeled cases in `data/cases.json`, once per
prompt version, and reports:

- Tier agreement (predicted vs. gold `urgency_tier`)
- **Critical false-negative rate** — gold is Critical but the prediction is lower;
  the failure mode that matters most, since plain accuracy hides it
- Average token cost per case
- How often the fallback provider was invoked, and what that did to total run cost
- A v1 → v2 delta table, with a regression warning if v2 increases the Critical
  false-negative rate relative to v1

Per-run audit trails are written to `output/audit_log_v1.jsonl` and `output/audit_log_v2.jsonl`.

## Provider configuration

Configured in `config/config.yaml` (loaded via `AppConfig.default()`), matching Part 4's
exact rate table:

| Provider | failure_rate | malformed_rate | $/1M in | $/1M out |
|---|---|---|---|---|
| orca-primary | 0.15 | 0.05 | 3.00 | 15.00 |
| orca-fallback | 0.02 | 0.25 | 0.50 | 1.50 |

`max_retries` (per-provider retry budget, exponential backoff), `backoff_base_seconds`, and
`max_schema_retries` (corrective re-prompt budget on invalid model output) are also set there.

## Tests

```bash
pytest
```

Covers: the retry/backoff/fallback cascade and cost/usage attribution (`test_llm_client.py`,
`test_fallback.py`), schema validation and the corrective-reprompt-then-fail-safe path
(`test_validation.py`), the non-negotiable Critical override and disallowed-language guardrail
(`test_guardrails.py`), and full end-to-end pipeline + audit-trail behavior (`test_triage.py`).

## Design decisions

See [`docs/design_decisions.md`](docs/design_decisions.md) for the Part 5 write-up: guardrail
enforcement strategy, fallback-provider trust, the asymmetric-risk eval metric, the v1/v2
regression gate call, and the cost-vs-safety retry cutoff.