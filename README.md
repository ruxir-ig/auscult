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
  no raw-storage mode.
- The sanitizer (`src/auscult/sanitizer.py`) uses Microsoft Presidio to detect
  PHI entities (names, phones, emails, dates/DOB, locations, street addresses,
  SSNs, MRNs, patient IDs) and Faker to substitute realistic synthetic values.
- Replacements are consistent within a run: the same real value always maps to
  the same fake value, so traces stay coherent for replay and debugging.

> **Caveat:** sanitization is detection-based. Anything Presidio and the custom
> recognizers miss is stored as-is. Treat the database as sensitive until you
> have validated detection quality on your own traffic.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (all dependencies, including the spaCy model,
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

`DATABASE_URL` is read lazily, only when a database connection is actually
needed, so `uv run auscult --help` works without it.

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
uv run auscult runs          # list all runs
uv run auscult run <id>      # one run with all of its steps
uv run auscult steps <id>    # just the steps of a run
```

Run the smoke test end to end:

```bash
uv run smoke_test.py
```

## Tests

```bash
uv run pytest
```

Tests cover PHI detection (names, phones, emails, dates/DOB formats, MRN
variants, bare patient IDs, street addresses, clinical free text), replacement
consistency, and a guarantee that raw PHI never reaches the database.

## Packaging note

The spaCy model `en_core_web_sm` is installed from a wheel URL declared in
`[tool.uv.sources]` in `pyproject.toml`. This is uv-specific: if you ever build
or install this package outside uv (plain pip, another resolver), that source
table is ignored and you must install the model yourself, e.g.
`python -m spacy download en_core_web_sm`.
