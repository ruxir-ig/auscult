# Auscult

Passive observability and safe replay tool for healthcare AI agents.

Auscult records what an AI agent did (prompt, steps, outputs, errors), sanitizes
all free text with PHI detection before anything touches the database, and makes
the resulting synthetic traces inspectable for debugging, QA, and audit — without
leaking patient data.

## How it works

- `AuscultTracer` (in `src/auscult/capture.py`) records a run and its steps.
- Every text field (`initial_prompt`, `llm_command`, `output`, `error_message`)
  passes through the sanitizer **before** it is written to the database. There is
  no raw-storage mode. If sanitization raises, the step is **not** written
  (fail-closed).
- The sanitizer (`src/auscult/sanitizer.py`) uses Microsoft Presidio to detect
  PHI entities (names, phones, emails, dates/DOB, locations, street addresses,
  SSNs, MRNs, patient IDs, org-specific badge/case IDs) and Faker to substitute
  realistic synthetic values.
- Replacements are consistent within a run: the same real value always maps to
  the same fake value, so traces stay coherent for replay and debugging.
- Each `Run` / `Step` stores a `redaction_count` so you can monitor detection
  volume and spot drift as agent output patterns change.
- JSON-shaped payloads are sanitized leaf-by-leaf so Faker replacements cannot
  corrupt object structure used later in replay comparisons.

> **Caveat:** sanitization is detection-based. Anything Presidio and the custom
> recognizers miss is stored as-is. Treat the database as sensitive until you
> have validated detection quality on your own traffic. "Sanitized" is not the
> same guarantee as "public."

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (all dependencies, including spaCy models,
  are managed through uv — never use pip directly)
- PostgreSQL (or SQLite for local experiments)

## Setup

```bash
uv sync
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost/auscult
# or, for a quick local run:
export DATABASE_URL=sqlite:////tmp/auscult.sqlite
uv run migrate.py
```

`migrate.py` applies Alembic migrations (`alembic upgrade head`). For schema
changes after the initial release, autogenerate a new revision:

```bash
uv run alembic revision --autogenerate -m "describe your change"
uv run alembic upgrade head
```

`DATABASE_URL` is read lazily for application code, and from the environment
when running migrations, so `uv run auscult --help` works without it.

### Sanitizer configuration

| Variable | Default | Meaning |
|---|---|---|
| `AUSCULT_SPACY_MODEL` | `en_core_web_lg` | spaCy model for NER. Use `en_core_web_trf` for best PERSON/LOCATION recall, or `en_core_web_sm` for faster local/test loads. |
| `AUSCULT_SCORE_THRESHOLD` | `0.35` | Minimum Presidio confidence to redact. **Lower = more false positives redacted** (safer for PHI; may over-redact clinical eponyms that slip past the allow-list). |

Clinical disease eponyms (Parkinson, Addison, Crohn, …) are allow-listed so they
are not treated as PERSON. Organization-specific patterns (e.g. `BADGE:…`,
`CASE#…`) are deny-listed as `CUSTOM_IDENTIFIER`. Pass extra patterns via
`Sanitizer(denylist_patterns=[...])`.

### Production guidance (defense in depth)

Even though the DB is meant to hold synthetic data:

1. **TLS to Postgres** — use `sslmode=require` (or verify-full) in `DATABASE_URL`.
2. **At-rest encryption** — enable volume encryption and/or column encryption for
   the Auscult database; sanitized ≠ public.
3. **Access control** — restrict who can `SELECT` from `runs` / `steps`; treat
   traces as PHI-adjacent until audited.
4. **Validate detection** — sample runs periodically; watch `redaction_count`
   drift via `auscult stats`.

## Usage

Instrument an agent:

```python
from auscult.capture import AuscultTracer

tracer = AuscultTracer(agent_type="triage-agent", initial_prompt=prompt)
tracer.record_step(llm_command=command, output=output, error_message=error)
tracer.finish()
```

Inspect traces from the CLI:

```bash
uv run auscult runs
uv run auscult runs --agent-type triage-agent --status failed --since 2026-07-01 --limit 50
uv run auscult run <id>
uv run auscult steps <id>
uv run auscult replay <id>
uv run auscult compare <id> --handler mypkg.handlers:my_agent_handler
uv run auscult stats
uv run auscult --json runs --limit 10    # machine-readable output for scripting
```

Replay a run in Python (for QA or regression checks):

```python
from auscult.replay import RunReplayer

replayer = RunReplayer.from_run_id(run_id)

# Playback: walk recorded steps without calling an agent
for step in replayer.playback():
    print(step.llm_command, "->", step.output or step.error_message)

# Compare: re-run your agent handler and diff against the recording
result = replayer.replay(my_agent_handler)  # handler(cmd) -> (output, error)
assert result.all_matched
```

## Tests

```bash
uv sync --group dev
AUSCULT_SPACY_MODEL=en_core_web_sm uv run pytest
uv run ruff check src tests
uv run mypy
```

Run the end-to-end integration test (capture, migrations, sanitization, replay):

```bash
uv run pytest tests/test_integration.py
# or
uv run smoke_test.py
```

Tests cover PHI detection (names, phones, emails, dates/DOB formats, MRN
variants, bare patient IDs, street addresses, clinical free text), clinical
allow-list / deny-list behavior, JSON payload structure preservation,
replacement consistency, fail-closed capture on sanitizer errors, and a
guarantee that raw PHI never reaches the database.

## Packaging note

spaCy models are installed from wheel URLs declared in `[tool.uv.sources]` in
`pyproject.toml` (`en_core_web_lg` by default, `en_core_web_sm` also packaged
for tests/dev). This is uv-specific: if you ever build or install this package
outside uv (plain pip, another resolver), that source table is ignored and you
must install the model yourself, e.g. `python -m spacy download en_core_web_lg`.
For `en_core_web_trf`, install the transformer model separately and set
`AUSCULT_SPACY_MODEL=en_core_web_trf`.
