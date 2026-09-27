# fake_llm.py — provided starter code.
# Do NOT call any real LLM API in this assignment. This module simulates
# one, deterministically, so your integration layer and evals are fully
# reproducible without an API key, network access, or token cost.

import random
import time
import json


class LLMError(Exception):
    pass


class LLMResponse:
    def __init__(self, text, input_tokens, output_tokens, provider):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.provider = provider


class FakeLLMProvider:
    """
    Simulates a single LLM provider/endpoint.

    failure_rate    -> probability the call raises LLMError (a transient
                        failure, like a real provider's 429/5xx).
    malformed_rate  -> probability the call "succeeds" but returns text
                        that is NOT valid JSON against the expected schema
                        (simulates a model ignoring formatting instructions).
    seed            -> fixes the RNG so repeated runs are reproducible.
    """

    def __init__(self, name, failure_rate=0.0, malformed_rate=0.0, seed=42):
        self.name = name
        self.failure_rate = failure_rate
        self.malformed_rate = malformed_rate
        self._rng = random.Random(seed)

    def complete(self, prompt: str, case_note: str) -> LLMResponse:
        time.sleep(0.02)  # simulated latency
        roll = self._rng.random()
        if roll < self.failure_rate:
            raise LLMError(f"{self.name}: simulated transient failure")

        malformed = self._rng.random() < self.malformed_rate
        text = _synthesize_response(case_note, malformed=malformed)
        input_tokens = max(20, len(prompt.split()))
        output_tokens = max(10, len(text.split()))
        return LLMResponse(text, input_tokens, output_tokens, self.name)


def _synthesize_response(case_note: str, malformed: bool = False) -> str:
    note = case_note.lower()

    if malformed:
        # Deliberately NOT valid JSON — your validation layer must catch this.
        return "Sure, here's the assessment: tier high, steps: contact caseworker"

    if "weapon" in note or "immediate danger" in note or "plan to harm" in note:
        tier = "Critical"
    elif "unsafe" in note or "missing" in note or "unsupervised" in note:
        tier = "High"
    elif "follow-up" in note or "check-in" in note or "documentation update" in note:
        tier = "Low"
    else:
        tier = "Medium"

    payload = {
        "urgency_tier": tier,
        "recommended_next_steps": [
            "Schedule caseworker contact within 24 hours",
            "Verify current safety and location",
        ],
        "flags": ["needs_review"] if tier in ("High", "Critical") else [],
        "confidence": 0.7,
    }
    return json.dumps(payload)


# A small, illustrative (NOT exhaustive) disallowed-terms list for the
# clinical/diagnostic-language guardrail in Part 2. In production this
# would be a much more rigorous taxonomy owned jointly with the functional
# lead — for this assignment, treat it as a stand-in.
DISALLOWED_TERMS = ["diagnose", "diagnosis", "prescribe", "psychiatric disorder", "medication dosage"]