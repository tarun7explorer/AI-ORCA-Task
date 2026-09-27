"""
ORCA AI Triage Layer — Web Console
===================================

A thin FastAPI wrapper around the existing ORCA triage pipeline
(`orca.triage.assistant.TriageCaseAssistant`) so a caseworker/reviewer can
exercise the real triage + guardrail + audit pipeline from a browser.

This module does NOT modify or reimplement any pipeline logic:
- fake_llm.py is untouched.
- src/orca/triage/* and src/orca/audit/* are imported and used as-is.
- src/orca/llm/* (LLMClient, FakeLLMProvider) are imported and used as-is.

The frontend (templates/index.html) is served as a plain static file — all
preset data lives client-side in JavaScript, so there is no server-side
template loop and nothing for a templating engine to mis-render.

Run locally:
    uvicorn app:app --reload --port 8000

Run in production (Render):
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — make `src/` importable regardless of the working directory
# the process is launched from (Linux / macOS / Windows all handled via
# pathlib, no hardcoded separators).
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEMPLATES_DIR = ROOT / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

from orca.audit.logger import AuditLogger  # noqa: E402
from orca.config import ProviderConfig  # noqa: E402
from orca.llm.client import LLMClient  # noqa: E402
from orca.llm.fake_llm import FakeLLMProvider  # noqa: E402
from orca.triage.assistant import TriageCaseAssistant  # noqa: E402

try:
    # Real guardrail engine — used verbatim to demonstrate the
    # disallowed-language guardrail firing on a synthetic example, since
    # the deterministic fake LLM never emits clinical language on its own.
    from orca.triage.guardrails import run_guardrails  # noqa: E402
except ImportError:  # pragma: no cover - defensive, keeps the console usable
    run_guardrails = None

app = FastAPI(title="ORCA AI Triage Layer", version="1.1.0")

# CORS enabled broadly — this is a demo console, not a multi-tenant
# production API, and open CORS avoids deployment-networking surprises
# (e.g. serving the page from one Render origin while calling the API
# from a preview URL or a different port during local development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Provider / client configuration — Part 4 rate table, wired verbatim.
# ---------------------------------------------------------------------------

PRIMARY_NAME = "orca-primary"
FALLBACK_NAME = "orca-fallback"

PRIMARY_FAILURE_RATE = 0.15
PRIMARY_MALFORMED_RATE = 0.05
PRIMARY_INPUT_RATE = 3.00
PRIMARY_OUTPUT_RATE = 15.00
PRIMARY_SEED = 42

FALLBACK_FAILURE_RATE = 0.02
FALLBACK_MALFORMED_RATE = 0.25
FALLBACK_INPUT_RATE = 0.50
FALLBACK_OUTPUT_RATE = 1.50
FALLBACK_SEED = 4242

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.05
MAX_SCHEMA_RETRIES = 2

PRIMARY_CFG = ProviderConfig(
    PRIMARY_NAME, PRIMARY_FAILURE_RATE, PRIMARY_MALFORMED_RATE,
    PRIMARY_INPUT_RATE, PRIMARY_OUTPUT_RATE, seed=PRIMARY_SEED,
)
FALLBACK_CFG = ProviderConfig(
    FALLBACK_NAME, FALLBACK_FAILURE_RATE, FALLBACK_MALFORMED_RATE,
    FALLBACK_INPUT_RATE, FALLBACK_OUTPUT_RATE, seed=FALLBACK_SEED,
)

_primary_provider = FakeLLMProvider(
    PRIMARY_NAME, PRIMARY_FAILURE_RATE, PRIMARY_MALFORMED_RATE, seed=PRIMARY_SEED
)
_fallback_provider = FakeLLMProvider(
    FALLBACK_NAME, FALLBACK_FAILURE_RATE, FALLBACK_MALFORMED_RATE, seed=FALLBACK_SEED
)

# One long-lived client per process so cumulative usage/cost tracking
# behaves the same way it would in a real deployment.
CLIENT = LLMClient(
    primary=_primary_provider,
    fallback=_fallback_provider,
    primary_cfg=PRIMARY_CFG,
    fallback_cfg=FALLBACK_CFG,
    max_retries=MAX_RETRIES,
    backoff_base_seconds=BACKOFF_BASE_SECONDS,
)

# ---------------------------------------------------------------------------
# Prompt templates — loaded from prompts/ on disk (Part 3's v1 / v2 files),
# with an embedded fallback so the app still boots if the files are moved.
# ---------------------------------------------------------------------------

PROMPTS_DIR = ROOT / "prompts"

_DEFAULT_V1 = (
    "You are the ORCA triage assistant for Solace Relief Network.\n\n"
    "Case ID: {case_id}\n"
    "Case note: {case_note}\n\n"
    "Assess the urgency of this case and respond with ONLY a single JSON object,\n"
    "no extra text, matching this schema exactly:\n\n"
    '{{"urgency_tier": "Low|Medium|High|Critical", "recommended_next_steps": ["..."], '
    '"flags": ["..."], "confidence": 0.0-1.0}}'
)

_DEFAULT_V2 = (
    "ORCA Case Triage — Structured Output Required\n\n"
    "Read the case note below and think step by step about safety risk before\n"
    "answering. Consider: immediate physical danger, self-harm risk, unverified\n"
    "vs. confirmed reports, and time since last contact.\n\n"
    "Case ID: {case_id}\n"
    "Case note: {case_note}\n\n"
    "Required urgency tiers, in increasing severity: Low, Medium, High, Critical.\n"
    "Output ONLY one JSON object (no prose, no markdown fences) with these exact\n"
    "keys: urgency_tier (one of the four tiers above), recommended_next_steps\n"
    "(array of short action strings), flags (array of strings from your internal\n"
    "vocabulary), confidence (a float between 0.0 and 1.0 reflecting how certain\n"
    "you are in this tier assignment given the information available)."
)


def _load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"prompt_{version}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return _DEFAULT_V1 if version == "v1" else _DEFAULT_V2


PROMPTS = {"v1": _load_prompt("v1"), "v2": _load_prompt("v2")}

# Illustrative disallowed phrase used ONLY to demonstrate the real guardrail
# engine firing. The deterministic fake LLM never emits clinical language on
# its own (see fake_llm._synthesize_response), so the "clinical_language"
# preset appends this to a genuinely-produced record and re-runs the actual
# guardrail function against it — the guardrail logic itself is untouched.
DISALLOWED_DEMO_PHRASE = (
    "Caseworker should diagnose the client's psychiatric disorder before "
    "scheduling a follow-up and consider medication dosage adjustments."
)


class TriageRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=4000)
    prompt_version: str = Field(default="v1")
    preset: Optional[str] = None


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _record_to_dict(record: Any, case_id: str) -> dict:
    if hasattr(record, "to_dict"):
        try:
            return record.to_dict()
        except Exception:
            pass
    return {
        "case_id": case_id,
        "urgency_tier": _safe_getattr(record, "urgency_tier"),
        "recommended_next_steps": list(_safe_getattr(record, "recommended_next_steps", []) or []),
        "flags": list(_safe_getattr(record, "flags", []) or []),
        "confidence": _safe_getattr(record, "confidence"),
        "requires_human_review": _safe_getattr(record, "requires_human_review"),
    }


@app.get("/")
async def index():
    return FileResponse(str(INDEX_HTML))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/api/triage")
async def api_triage(payload: TriageRequest):
    case_id = f"WEB-{uuid.uuid4().hex[:8].upper()}"
    version = payload.prompt_version if payload.prompt_version in PROMPTS else "v1"
    prompt_template = PROMPTS[version]

    # A fresh audit logger per request keeps concurrent requests from
    # racing on "the last logged record" while still exercising the real
    # AuditLogger implementation for every call.
    audit_logger = AuditLogger(output_path=None)
    assistant = TriageCaseAssistant(
        client=CLIENT,
        prompt_template=prompt_template,
        audit_logger=audit_logger,
        max_schema_retries=MAX_SCHEMA_RETRIES,
    )

    try:
        started = time.perf_counter()
        record = assistant.triage(case_id, payload.note)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Triage pipeline raised an unexpected error.",
                "error": str(exc),
                "trace": traceback.format_exc(limit=5),
            },
        )

    demo_injected = False
    if payload.preset == "clinical_language":
        try:
            record.recommended_next_steps.append(DISALLOWED_DEMO_PHRASE)
            if run_guardrails is not None:
                result = run_guardrails(record)
                record = result[0] if isinstance(result, tuple) else result
            else:
                if "disallowed_clinical_language" not in record.flags:
                    record.flags.append("disallowed_clinical_language")
                record.requires_human_review = True
            demo_injected = True
        except Exception:
            demo_injected = False

    audit_records = audit_logger.all_records()
    audit = audit_records[-1] if audit_records else None

    guardrail_intervened = bool(_safe_getattr(audit, "guardrail_intervened", False)) or demo_injected

    response = {
        "case_id": case_id,
        "prompt_version": version,
        "urgency_tier": _safe_getattr(record, "urgency_tier"),
        "requires_human_review": bool(_safe_getattr(record, "requires_human_review", False)),
        "confidence": _safe_getattr(record, "confidence"),
        "flags": list(_safe_getattr(record, "flags", []) or []),
        "recommended_next_steps": list(_safe_getattr(record, "recommended_next_steps", []) or []),
        "provider_used": _safe_getattr(audit, "provider_used"),
        "attempts": _safe_getattr(audit, "attempts"),
        "schema_retried": bool(_safe_getattr(audit, "schema_retried", False)),
        "guardrail_intervened": guardrail_intervened,
        "fail_safe": bool(_safe_getattr(audit, "fail_safe", False)),
        "cost_usd": _safe_getattr(audit, "cost_usd"),
        "latency_ms": latency_ms,
        "timestamp": _safe_getattr(audit, "timestamp"),
        "raw_model_output": _safe_getattr(audit, "raw_model_output"),
        "validated_output": _record_to_dict(record, case_id),
        "demo_injected": demo_injected,
    }
    return JSONResponse(response)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)