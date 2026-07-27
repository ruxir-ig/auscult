# Auscult

Python SDK and CLI for PHI-safe observability and replay of healthcare AI agents.

Auscult records what an AI agent did (prompt, steps, outputs, errors), sanitizes
all free text with PHI detection before anything touches the database, and makes
the resulting synthetic traces inspectable for debugging, QA, and audit — without
leaking patient data.

## Install

**With uv (recommended):**

```bash
uv add auscult
uv run auscult setup                 # downloads en_core_web_lg (PHI NER)
export DATABASE_URL=sqlite:////tmp/auscult.sqlite   # or Postgres
uv run auscult migrate
```

**With pip:**

```bash
pip install auscult
auscult setup
export DATABASE_URL=sqlite:////tmp/auscult.sqlite
auscult migrate
```

LangChain / LangGraph support is optional:

```bash
uv add 'auscult[langchain]'
# or: pip install 'auscult[langchain]'
```

> **Note:** spaCy models are not on PyPI, so they are installed separately via
> `auscult setup` (or `python -m spacy download en_core_web_lg`). PyPI rejects
> packages that declare direct URL dependencies.

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
- Each `Run` / `Step` stores a `redaction_count` and `entity_counts` (entity
  type → count, no raw spans) so you can monitor detection volume and mix.
- JSON-shaped payloads are sanitized leaf-by-leaf so Faker replacements cannot
  corrupt object structure used later in replay comparisons.
- Faker is seeded from the run id, so re-capturing the same raw input with the
  same `run_id` yields identical synthetic replacements.
- Optional `background=True` moves sanitize + DB I/O onto a worker thread so
  the agent hot path stays passive; a full queue or worker error fails closed.

> **Caveat:** sanitization is detection-based. Anything Presidio and the custom
> recognizers miss is stored as-is. Treat the database as sensitive until you
> have validated detection quality on your own traffic. "Sanitized" is not the
> same guarantee as "public."

## Requirements

- Python 3.12+
- PostgreSQL (or SQLite for local experiments)
- A spaCy English model (`auscult setup` installs `en_core_web_lg` by default)

## Quickstart

```bash
uv add auscult
uv run auscult setup
export DATABASE_URL=sqlite:////tmp/auscult.sqlite
uv run auscult migrate
```

### Plug into an existing agent (no loop rewrite)

Most agents don't need explicit `record_step` calls. Wrap the LLM client (or
pass a callback handler) once and every model call is captured automatically:

```python
from openai import OpenAI
from auscult import observe_run
from auscult.integrations.openai import wrap_openai

client = wrap_openai(OpenAI())   # also: Azure / any OpenAI-compatible server

@observe_run(agent_type="triage-agent")
def run_triage(prompt: str) -> str:
    # existing agent loop, unchanged — every chat.completions.create /
    # responses.create call is sanitized and captured as a step
    return my_existing_agent(client, prompt)
```

Anthropic works the same way via
`auscult.integrations.anthropic.wrap_anthropic(client)` (patches
`messages.create`).

For LangChain / LangGraph (install with `auscult[langchain]`), pass a
callback handler at invocation — the standard pluggable observability
mechanism for those frameworks:

```python
from auscult.integrations.langchain import AuscultCallbackHandler

handler = AuscultCallbackHandler(agent_type="triage-agent")
result = graph.invoke(inputs, config={"callbacks": [handler]})
print(handler.last_run_id)   # inspect with: auscult run <id>
```

Each LLM call and tool call becomes one sanitized step; the run finishes when
the root chain/graph ends (marked `failed` on error). The handler sets
`raise_error`, so sanitizer/DB failures propagate (fail-closed) instead of
being swallowed by LangChain.

If you prefer explicit context, `start_run` activates an ambient run that all
integrations record into (it propagates through `asyncio`, but not into
`ThreadPoolExecutor` workers — pass `tracer=` explicitly there):

```python
from auscult import start_run, record_step

with start_run("triage-agent", initial_prompt=prompt):
    run_existing_agent(client, prompt)   # wrapped-client calls captured
    record_step("manual_annotation()", output="anything else you want traced")
```

### Instrument by hand

The explicit API is still there for custom loops:

```python
from auscult.capture import AuscultTracer

tracer = AuscultTracer(agent_type="triage-agent", initial_prompt=prompt)
tracer.record_step(llm_command=command, output=output, error_message=error)
tracer.finish()

# Non-blocking capture for high-frequency agents (sanitize+DB on a worker):
with AuscultTracer(
    agent_type="triage-agent",
    initial_prompt=prompt,
    background=True,
) as tracer:
    tracer.record_step(llm_command=command, output=output)
```

### Inspect from the CLI

```bash
auscult runs
auscult runs --agent-type triage-agent --status failed --since 2026-07-01 --limit 50
auscult run <id>
auscult steps <id>
auscult replay <id>
auscult compare <id> --handler mypkg.handlers:my_agent_handler
auscult stats
auscult export <id> --format jsonl -o run.jsonl
auscult purge --before 2026-01-01 --dry-run
auscult eval --verbose
auscult eval --dual-pass-model en_core_web_trf
auscult --json runs --limit 10    # machine-readable output for scripting
```

See [ROADMAP.md](ROADMAP.md) for later ideas (audit UI, per-tenant lists, metrics).

## Developing from source

```bash
git clone https://github.com/ruxir-ig/auscult.git
cd auscult
uv sync --group nlp          # package + en_core_web_lg via uv sources
# or, for tests/CI (small model):
uv sync --group dev
export DATABASE_URL=sqlite:////tmp/auscult.sqlite
uv run auscult migrate
# equivalent: uv run migrate.py
```

Schema revisions (contributors):

```bash
uv run alembic revision --autogenerate -m "describe your change"
uv run auscult migrate
```

`DATABASE_URL` is read lazily for application code, and from the environment
when running migrations, so `auscult --help` works without it.

### Sanitizer configuration

| Variable | Default | Meaning |
|---|---|---|
| `AUSCULT_SPACY_MODEL` | `en_core_web_lg` | Primary spaCy model for NER. Use `en_core_web_trf` for best PERSON/LOCATION recall, or `en_core_web_sm` for faster local/test loads (`auscult setup --model …`, or `uv sync --group dev` / `--group nlp-trf`). |
| `AUSCULT_SCORE_THRESHOLD` | `0.35` | Minimum Presidio confidence to redact. **Lower = more false positives redacted** (safer for PHI; may over-redact clinical eponyms that slip past the allow-list). |
| `AUSCULT_DUAL_PASS_MODEL` | _(unset)_ | Optional second spaCy model (e.g. `en_core_web_trf`). When set, PERSON/LOCATION candidates from this model are merged with the primary pass for better recall without running the heavy model on every entity type. |

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
4. **Validate detection** — run `auscult eval` on the golden corpus after model /
   threshold changes; watch `redaction_count` / `entity_counts` drift via
   `auscult stats`.

### Indexes

Migrations create composite indexes for filtered list/stats queries:

- `runs(agent_type, started_at)`, `runs(status, started_at)`
- unique `(run_id, step_index)` on `steps`

### Replay in Python

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
uv run auscult eval --nlp-model en_core_web_sm
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
replacement consistency, fail-closed capture on sanitizer errors, the golden
eval corpus, dual-pass merge behavior, and a guarantee that raw PHI never
reaches the database.

## Packaging note

spaCy models are **not** declared as package dependencies (PyPI forbids direct
URL requirements). After `pip` / `uv` install:

```bash
auscult setup                              # en_core_web_lg
auscult setup --model en_core_web_sm       # smaller / faster
auscult setup --model en_core_web_trf      # best PERSON recall
```

From a source checkout, uv can pull model wheels via `[tool.uv.sources]`:

- `uv sync --group nlp` — `en_core_web_lg`
- `uv sync --group nlp-trf` — `en_core_web_trf`
- `uv sync --group dev` — `en_core_web_sm` + test tools + `langchain-core`

Optional PyPI extra:

- `auscult[langchain]` — `langchain-core` for the LangChain/LangGraph callback
  handler (OpenAI/Anthropic wrappers are duck-typed and need no extra)
