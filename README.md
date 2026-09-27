Live Link - [https://ai-orca-task.onrender.com/]

# ORCA AI Layer

Triage assistant, guardrails, evaluation harness, and web console for SRN's ORCA
case-management platform.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Quickstart

### Interactive web console

Launch the FastAPI-based triage console locally:

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser. The console lets
you pick a prompt version (v1 / v2), load one of four quick-scenario presets, submit a
case note, and inspect the resulting urgency tier, human-review status, audit metadata
(provider used, attempts, schema retries, call cost), and raw JSON payload — all served
by the same `TriageCaseAssistant` pipeline used by the CLI and eval harness below.

The API endpoint backing the console is `POST /api/triage`, which accepts:

```json
{ "note": "Case note text...", "prompt_version": "v1" }
```

### Run a single case (CLI)

```bash
python scripts/triage_case.py --case-id CASE-123 \
  --note "Client reported their partner has access to a weapon and recent threats were made." \
  --prompt-version v1
```

### Run the eval harness (v1 vs v2 regression check)

```bash
python eval/run_eval.py
```

Reports tier agreement, Critical false-negative rate, avg token cost/case, and fallback-usage
rate for both prompt versions, plus a v1→v2 delta table.

## Provider configuration

Configured in `config/config.yaml` (or `AppConfig.default()` in code) per Part 4's rate table:

| Provider | failure_rate | malformed_rate | $/1M in | $/1M out |
|---|---|---|---|---|
| orca-primary | 0.15 | 0.05 | 3.00 | 15.00 |
| orca-fallback | 0.02 | 0.25 | 0.50 | 1.50 |

## Deployment

The repo includes a Render Blueprint (`render.yaml`) for one-click deployment of the web
console on Render's free tier:

1. Push this repository to GitHub (or GitLab).
2. In Render, choose **New → Blueprint** and point it at the repo. Render reads
   `render.yaml` automatically and provisions a `web` service named `orca-ai-triage`
   with:
   - `buildCommand: pip install -r requirements.txt`
   - `startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT`
   - `PYTHONPATH=src` set so the `orca` package under `src/` resolves correctly.
3. Once deployed, the console is reachable at the Render-issued URL, and `/healthz`
   can be used as an uptime/health check.

No environment variables or secrets are required — the app runs entirely against the
deterministic `FakeLLMProvider` from Appendix B, so there's no API key to configure.

## Tests

```bash
pytest
```
