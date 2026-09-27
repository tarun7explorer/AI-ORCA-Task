"""
Part 1 — Thin LLM integration layer for the ORCA triage assistant.

Wraps a primary and fallback FakeLLMProvider (see orca.llm.fake_llm) behind
a single call-site interface:

    client.complete(prompt, case_note, feature="triage")

Behavior:
  - Retry logic on the primary: on a transient LLMError, retry with
    exponential backoff up to `max_retries` attempts before falling back
    to the secondary provider.
  - The fallback is retried under the same policy (up to `max_retries`
    attempts) before the call is considered exhausted.
  - Per-call and cumulative token usage (input + output) and dollar cost
    are tracked, attributable to a "feature" or "session" identifier.
  - "Success" here means the provider call returned without raising
    LLMError. Malformed (non-JSON) responses are NOT retried at this
    layer — that is a Part 2 concern (schema validation / re-prompting
    in the triage assistant). This client's only job is to get *a*
    response back from *a* provider and report which one, honestly.

Documented fail-safe path:
  If both the primary and the fallback exhaust their retry budget without
  a single successful (non-LLMError) response, `complete()` does NOT
  raise. It returns a `CompletionResult` with `fail_safe=True`,
  `text=None`, `provider_used=None`, `cost_usd=0.0`, and `error` set to a
  human-readable description of the last failure. Callers (the triage
  assistant) are expected to check `fail_safe` and substitute their own
  safe default rather than propagate a raw exception to a caseworker-
  facing surface.

This module never modifies orca.llm.fake_llm — it only consumes it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from orca.config import ProviderConfig
from orca.llm.fake_llm import FakeLLMProvider, LLMError


@dataclass
class CompletionResult:
    """Structured result returned by LLMClient.complete()."""

    text: Optional[str]
    provider_used: Optional[str]
    used_fallback: bool
    fail_safe: bool
    attempts: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None


@dataclass
class _FeatureUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    provider_counts: Dict[str, int] = field(default_factory=dict)


class UsageTracker:
    """Cumulative token/cost usage, keyed by feature/session identifier."""

    def __init__(self) -> None:
        self._features: Dict[str, _FeatureUsage] = {}

    def record(
        self,
        feature: str,
        provider_name: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        usage = self._features.setdefault(feature, _FeatureUsage())
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.total_cost += cost
        usage.provider_counts[provider_name] = usage.provider_counts.get(provider_name, 0) + 1

    def total_tokens(self, feature: Optional[str] = None) -> int:
        if feature is not None:
            usage = self._features.get(feature)
            return (usage.input_tokens + usage.output_tokens) if usage else 0
        return sum(u.input_tokens + u.output_tokens for u in self._features.values())

    def total_cost(self, feature: Optional[str] = None) -> float:
        if feature is not None:
            usage = self._features.get(feature)
            return usage.total_cost if usage else 0.0
        return sum(u.total_cost for u in self._features.values())

    def provider_call_counts(self, feature: str) -> Dict[str, int]:
        usage = self._features.get(feature)
        return dict(usage.provider_counts) if usage else {}


class LLMClient:
    """
    Call-site interface: client.complete(prompt, case_note, feature="triage")

    Retry / fallback policy:
      1. Call the primary provider up to `max_retries` times. Each
         transient LLMError triggers an exponential backoff sleep
         (`backoff_base_seconds * 2**(attempt-1)`) before the next
         attempt.
      2. If the primary never succeeds, switch to the fallback provider
         for up to `max_retries` attempts, under the same backoff policy.
      3. If the fallback also never succeeds, return the documented
         fail-safe CompletionResult (see module docstring) rather than
         raising.
    """

    def __init__(
        self,
        primary: FakeLLMProvider,
        fallback: FakeLLMProvider,
        primary_cfg: ProviderConfig,
        fallback_cfg: ProviderConfig,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_cfg = primary_cfg
        self.fallback_cfg = fallback_cfg
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.usage = UsageTracker()

    @staticmethod
    def _cost(input_tokens: int, output_tokens: int, cfg: ProviderConfig) -> float:
        return (input_tokens / 1_000_000) * cfg.input_rate_per_1m + (
            output_tokens / 1_000_000
        ) * cfg.output_rate_per_1m

    def _try_provider(
        self,
        provider: FakeLLMProvider,
        cfg: ProviderConfig,
        prompt: str,
        case_note: str,
        feature: str,
        attempts_counter: list,
    ):
        """
        Attempt a single provider up to `self.max_retries` times, with
        exponential backoff between failures. Returns
        (LLMResponse, cost_usd) on the first non-error call, or
        (None, last_error_message) if every attempt raised LLMError.
        """
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            attempts_counter[0] += 1
            try:
                response = provider.complete(prompt, case_note)
            except LLMError as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    backoff_seconds = self.backoff_base_seconds * (2 ** (attempt - 1))
                    if backoff_seconds > 0:
                        time.sleep(backoff_seconds)
                continue

            cost = self._cost(response.input_tokens, response.output_tokens, cfg)
            self.usage.record(
                feature, response.provider, response.input_tokens, response.output_tokens, cost
            )
            return response, cost

        return None, last_error

    def complete(self, prompt: str, case_note: str, feature: str = "triage") -> CompletionResult:
        """
        Attempt the primary provider first, then the fallback, per the
        retry/backoff policy documented on the class. Never raises for a
        provider failure — see the fail-safe path in the module docstring.
        """
        attempts_counter = [0]

        response, cost_or_error = self._try_provider(
            self.primary, self.primary_cfg, prompt, case_note, feature, attempts_counter
        )
        if response is not None:
            return CompletionResult(
                text=response.text,
                provider_used=response.provider,
                used_fallback=False,
                fail_safe=False,
                attempts=attempts_counter[0],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost_or_error,
                error=None,
            )
        primary_error = cost_or_error

        response, cost_or_error = self._try_provider(
            self.fallback, self.fallback_cfg, prompt, case_note, feature, attempts_counter
        )
        if response is not None:
            return CompletionResult(
                text=response.text,
                provider_used=response.provider,
                used_fallback=True,
                fail_safe=False,
                attempts=attempts_counter[0],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost_or_error,
                error=None,
            )
        fallback_error = cost_or_error

        # Both providers exhausted their retry budget: documented fail-safe path.
        error_message = (
            f"providers_exhausted: primary={primary_error!r} fallback={fallback_error!r}"
        )
        return CompletionResult(
            text=None,
            provider_used=None,
            used_fallback=True,
            fail_safe=True,
            attempts=attempts_counter[0],
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error=error_message,
        )