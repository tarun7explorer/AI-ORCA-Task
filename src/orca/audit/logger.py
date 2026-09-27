"""
Part 2 — Audit trail.

Every triage call produces one `AuditRecord`, capturing: the input note
id, the raw model output (None if the client itself was exhausted before
any text was produced), the validated/final output after guardrails, which
provider actually served the result, total attempts, whether a schema
repair retry happened, whether a guardrail intervened (and why), whether
the fail-safe path was taken, total dollar cost, total tokens, and an
ISO-8601 UTC timestamp.

`AuditLogger` keeps every record in memory (`all_records()`) and, if given
an `output_path`, additionally appends each record as a JSON line to that
file — so a run's full audit trail survives the process exiting.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class AuditRecord:
    case_id: str
    timestamp: str
    raw_model_output: Optional[str]
    validated_output: Optional[dict]
    provider_used: Optional[str]
    attempts: int
    schema_retried: bool
    guardrail_intervened: bool
    guardrail_reasons: List[str]
    fail_safe: bool
    cost_usd: float
    tokens: int = 0


class AuditLogger:
    """
    Thread-safe in-memory + optional append-only JSONL audit log.

    Pass `output_path=None` (e.g. in tests) to keep everything in memory
    only. Pass a path to also persist every record as it's logged.
    """

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = Path(output_path) if output_path else None
        self._records: List[AuditRecord] = []
        self._lock = threading.Lock()
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: AuditRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self.output_path:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record)) + "\n")

    def all_records(self) -> List[AuditRecord]:
        return list(self._records)


def new_audit_record(
    case_id: str,
    raw_model_output: Optional[str],
    validated_output: Optional[dict],
    provider_used: Optional[str],
    attempts: int,
    schema_retried: bool,
    guardrail_intervened: bool,
    guardrail_reasons: List[str],
    fail_safe: bool,
    cost_usd: float,
    tokens: int = 0,
) -> AuditRecord:
    return AuditRecord(
        case_id=case_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_model_output=raw_model_output,
        validated_output=validated_output,
        provider_used=provider_used,
        attempts=attempts,
        schema_retried=schema_retried,
        guardrail_intervened=guardrail_intervened,
        guardrail_reasons=guardrail_reasons,
        fail_safe=fail_safe,
        cost_usd=cost_usd,
        tokens=tokens,
    )