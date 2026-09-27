# ORCA Triage Layer — Design Decisions Brief

## 1. Guardrail enforcement strategy

Implemented: **pass-through, flagged for mandatory human review**
(`apply_disallowed_language_guardrail` in `guardrails.py`). Stripping the
offending text risks silently discarding a genuinely urgent signal (a step
mentioning "psychiatric disorder" might be pointing at something real);
reject-and-retry risks looping indefinitely if the model consistently
reaches for clinical framing for a legitimately severe case, and delays a
response that a caseworker may need quickly. Pass-through-with-flag keeps
the content, records that it was flagged, and forces a human into the loop
before anything clinical-sounding reaches a caseworker as an unreviewed
recommendation.

My answer does **not** differ between Critical and Low tiers on which
strategy to use — the strategy itself (flag + require review) is uniform.
What differs is consequence: on a Low-tier case this guardrail is often the
*only* reason a human ever looks at it, since Low doesn't otherwise trigger
review. On a Critical case, `requires_human_review` is already forced True
by the non-negotiable override, so the guardrail's practical effect is
smaller — but it still matters for the audit trail (`disallowed_clinical_language`
flag) and for surfacing the failure mode in eval/monitoring, independent of
tier.

## 2. Fallback provider trust

No — a result served by the fallback is **not** treated identically
downstream. The audit record captures `provider_used` for every call, so
which provider served a result is always visible and is never silently
lost. Given the fallback's much higher `malformed_rate` (0.25 vs 0.05), a
malformed fallback response is already caught by schema validation and
corrective re-prompting — that part is symmetric. What's asymmetric is
residual trust in a response that *did* parse cleanly: the fallback is
simply a weaker model/pipeline standing in for the primary, so I do not add
an additional blanket "fallback ⇒ always require review" rule, because that
would make `requires_human_review` a proxy for "which provider answered"
rather than for actual case severity, and would flood caseworkers with
low-value reviews on the ~1-in-5 calls where the primary is down. Instead,
provider identity is preserved in the audit log so it can be used for
monitoring (e.g. "are Critical predictions from the fallback disagreeing
with gold more than primary's?") without polluting the per-case decision
logic itself.

## 3. Eval metric choice under asymmetric risk

Beyond plain tier agreement, the harness computes **Critical false-negative
rate** specifically: among gold-Critical cases, the fraction where the
prediction was *not* Critical (`metrics.py::compute_metrics`). This isolates
the one failure direction that carries real safety risk — predicting
Critical when gold is Low is wasteful but recoverable via human review;
predicting non-Critical when gold is Critical can mean a genuinely
dangerous case gets routine handling. Plain accuracy would let a model
that's very good at Low/Medium cases mask a systematic blind spot on
Critical cases; this metric can't be averaged away.

## 4. Regression gate

If v2 improved overall agreement by 4% but doubled the Critical
false-negative rate, my call is **no-go, do not ship v2 as-is.** Overall
agreement is dominated by the more common, lower-stakes tiers (Low/Medium
in this dataset), so a 4% overall gain can be bought entirely by getting
easy cases slightly more right while getting the rare, highest-stakes cases
meaningfully more wrong — and the failure mode that matters most here isn't
"lower average error," it's "how often does a genuinely Critical case get
waved through as routine." A doubled false-negative rate on Critical is a
regression in exactly the dimension this system exists to protect against.
That said, I wouldn't discard v2 outright: the gap suggests v2's phrasing
is trading recall on severe cases for precision on trivial ones, which is
worth diagnosing (e.g. does v2's "think step by step" framing make the
model second-guess strong safety signals down to Medium?). The actual gate
I'd apply: no version ships if it increases the Critical false-negative
rate versus the current production prompt, full stop, regardless of any
gain elsewhere — that's the one number this system cannot regress on.

## 5. Cost vs. safety trade-off

Implemented: **one corrective re-prompt maximum** (`max_schema_retries = 1`
in `config/config.yaml` / `AppConfig` / `TriageCaseAssistant`), then fail
safe. Concrete rule: if the first response fails schema validation, retry
exactly once with a corrective instruction; if that also fails, stop —
don't retry a third time. Rationale for stopping at 1, not 0 or 3+: a
single re-prompt catches the common case (a `malformed_rate`-style
formatting slip) cheaply, but a model that fails validation twice in a row
on the same note is unlikely to self-correct on a third identical nudge,
and every extra attempt adds latency and cost for a caseworker waiting on
triage. "Failing safe" returns a synthetic `TriageRecord` with
`urgency_tier="High"` (deliberately *not* "Low," so it doesn't get silently
deprioritized), `requires_human_review=True`, `flags=["automation_failure"]`,
and a `recommended_next_steps` entry that says outright to escalate to a
human caseworker because automated triage was unavailable or invalid — the
caseworker sees an honest "the system couldn't triage this, look at it
yourself," not a fabricated confident answer.