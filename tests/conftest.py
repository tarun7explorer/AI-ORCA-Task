from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orca.audit.logger import AuditLogger
from orca.config import AppConfig, ProviderConfig
from orca.llm.client import LLMClient
from orca.llm.fake_llm import FakeLLMProvider
from orca.triage.assistant import TriageCaseAssistant

PROMPT_TEMPLATE = (
    "You are the ORCA triage assistant. Case ID: {case_id}\n"
    "Case note: {case_note}\n"
    "Respond with ONLY JSON: "
    '{{"urgency_tier": "Low|Medium|High|Critical", '
    '"recommended_next_steps": ["..."], "flags": ["..."], "confidence": 0.0-1.0}}'
)


@pytest.fixture
def prompt_template() -> str:
    return PROMPT_TEMPLATE


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig(
        primary=ProviderConfig(
            "orca-primary", failure_rate=0.15, malformed_rate=0.05,
            input_rate_per_1m=3.00, output_rate_per_1m=15.00, seed=42,
        ),
        fallback=ProviderConfig(
            "orca-fallback", failure_rate=0.02, malformed_rate=0.25,
            input_rate_per_1m=0.50, output_rate_per_1m=1.50, seed=4242,
        ),
        max_retries=3,
        backoff_base_seconds=0.0,
        max_schema_retries=1,
    )


def make_client(
    primary_failure: float = 0.0,
    primary_malformed: float = 0.0,
    fallback_failure: float = 0.0,
    fallback_malformed: float = 0.0,
    max_retries: int = 3,
    backoff: float = 0.0,
) -> LLMClient:
    primary_cfg = ProviderConfig("orca-primary", primary_failure, primary_malformed, 3.00, 15.00, seed=1)
    fallback_cfg = ProviderConfig("orca-fallback", fallback_failure, fallback_malformed, 0.50, 1.50, seed=2)
    primary = FakeLLMProvider("orca-primary", primary_failure, primary_malformed, seed=1)
    fallback = FakeLLMProvider("orca-fallback", fallback_failure, fallback_malformed, seed=2)
    return LLMClient(
        primary=primary,
        fallback=fallback,
        primary_cfg=primary_cfg,
        fallback_cfg=fallback_cfg,
        max_retries=max_retries,
        backoff_base_seconds=backoff,
    )


@pytest.fixture
def clean_client() -> LLMClient:
    """Primary always succeeds and always returns well-formed JSON."""
    return make_client(primary_failure=0.0, primary_malformed=0.0)


@pytest.fixture
def always_fail_primary_client() -> LLMClient:
    """Primary always fails; fallback always succeeds cleanly."""
    return make_client(
        primary_failure=1.0, primary_malformed=0.0,
        fallback_failure=0.0, fallback_malformed=0.0,
        max_retries=2,
    )


@pytest.fixture
def both_exhausted_client() -> LLMClient:
    """Both primary and fallback always fail -> fail-safe path."""
    return make_client(primary_failure=1.0, fallback_failure=1.0, max_retries=2)


@pytest.fixture
def always_malformed_client() -> LLMClient:
    """Primary always succeeds but always returns malformed (non-JSON) text."""
    return make_client(
        primary_failure=0.0, primary_malformed=1.0,
        fallback_failure=0.0, fallback_malformed=1.0,
    )


@pytest.fixture
def sample_case_notes():
    return [
        {"case_id": "case-01", "note": "Routine follow-up call completed. Client doing well, no concerns raised.", "gold_tier": "Low"},
        {"case_id": "case-02", "note": "Client missed last two scheduled check-ins. Neighbor reports client has not been seen in 5 days.", "gold_tier": "High"},
        {"case_id": "case-03", "note": "Client described feeling hopeless and mentioned a specific plan to harm themselves during the session.", "gold_tier": "Critical"},
        {"case_id": "case-06", "note": "Client reported their partner has access to a weapon and recent threats were made.", "gold_tier": "Critical"},
    ]


@pytest.fixture
def audit_logger() -> AuditLogger:
    return AuditLogger(output_path=None)


@pytest.fixture
def assistant_factory(prompt_template):
    def _make(client, max_schema_retries: int = 1, audit_logger: AuditLogger | None = None):
        return TriageCaseAssistant(
            client=client,
            prompt_template=prompt_template,
            audit_logger=audit_logger or AuditLogger(output_path=None),
            max_schema_retries=max_schema_retries,
        )

    return _make