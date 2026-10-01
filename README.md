# Auscult

PHI-aware observability and replay for healthcare AI agents.

Auscult captures agent prompts, model calls, outputs, and errors. It detects and replaces likely protected health information (PHI) before saving traces, so teams can inspect and replay runs for debugging and QA.

## Quickstart

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ruxir-ig/auscult.git
cd auscult
uv sync --group dev
uv run auscult setup --model en_core_web_sm
AUSCULT_SPACY_MODEL=en_core_web_sm uv run auscult demo
```

The demo creates a local SQLite database, applies migrations, and captures one run with synthetic example data. It prints the run ID and commands to inspect or replay that run. Set `DATABASE_URL` first if you want to use another database. `uv run python demo.py` does the same from a source checkout.

Use only synthetic data for demos.

Tried it? Clinicians and health-system staff can share what worked and what didn't through the [feedback form](https://github.com/ruxir-ig/auscult/issues/new?template=clinician_feedback.yml).

## Integrate with an agent

Wrap an OpenAI-compatible client and mark the agent entry point with `observe_run`:

```python
from openai import OpenAI
from auscult import observe_run
from auscult.integrations.openai import wrap_openai

client = wrap_openai(OpenAI())

@observe_run(agent_type="triage-agent")
def run_triage(prompt: str) -> str:
    return my_existing_agent(client, prompt)
```

Anthropic client wrapping, LangChain/LangGraph callbacks, and Codex CLI sessions are also supported. Install the optional LangChain dependency with `uv add 'auscult[langchain]'`. See [the integration examples](#integration-options) below.

## What Auscult captures

- Runs and ordered steps, including prompts, model commands, outputs, and errors.
- PHI detection and synthetic replacements before trace data is written.
- Redaction totals and counts by detected entity type.
- CLI tools to inspect, replay, compare, export, and summarize runs.
- Guards that sanitize files and tool results before an agent reads them.

Capture can also be used explicitly with `AuscultTracer`, `start_run`, and `record_step`. See [the API examples](#integration-options).

## Keep PHI out of agent context

Tracing sanitizes what Auscult stores. The guard sanitizes what the agent reads, so detected PHI is masked before it enters the model context.

In Python, sanitize file contents and tool results:

```python
from auscult.guard import guard_tool, read_sanitized

note = read_sanitized("notes/visit.txt")

@guard_tool
def fetch_chart(patient_id: str) -> dict:
    return ehr.get_chart(patient_id)  # str, dict, or list results are sanitized
```

From a shell or a coding agent, `auscult sanitize FILE` prints the masked text, and `auscult sanitize --check FILE` exits 1 when PHI is found.

For Claude Code, add a `PreToolUse` hook in `.claude/settings.json` that blocks reads of files that contain PHI. The block message tells the agent to read `auscult sanitize FILE` instead:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [{ "type": "command", "command": "auscult guard-hook" }]
      }
    ]
  }
}
```

The hook reads the file path from `tool_input.file_path` or `tool_input.path`. It blocks a file that cannot be scanned and allows paths that do not exist. It does not inspect shell commands, so pair it with sandbox or permission rules.

## Important limitation

PHI protection is detection-based. Detectors can miss sensitive information, so validate performance on representative data and treat stored traces as sensitive until your organization has approved their use. Do not use real patient data for a demo. Use `auscult eval` to measure detection precision and recall on the packaged corpus or your own.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | required | SQLAlchemy URL for trace storage. |
| `AUSCULT_SPACY_MODEL` | `en_core_web_lg` | spaCy model used for PHI detection. |
| `AUSCULT_SCORE_THRESHOLD` | `0.35` | Minimum detector confidence to redact. Lower redacts more. |
| `AUSCULT_DUAL_PASS_MODEL` | unset | Optional second spaCy model whose PERSON and LOCATION hits are merged in. |

`AuscultTracer` and `start_run` also accept `background=True` to sanitize and write steps on a worker thread, and `commit_each_step=False` to commit once when the run finishes.

## Development

```bash
uv sync --group dev
AUSCULT_SPACY_MODEL=en_core_web_sm uv run pytest
uv run ruff check src tests
uv run mypy
```

See [ROADMAP.md](ROADMAP.md) for additional project ideas.

## Integration options

### OpenAI-compatible clients

`wrap_openai` captures supported chat completions and Responses API calls. Azure and other OpenAI-compatible clients can also be wrapped.

### Anthropic

Use `auscult.integrations.anthropic.wrap_anthropic(client)` to capture `messages.create` calls.

### LangChain and LangGraph

Pass the callback handler at invocation:

```python
from auscult.integrations.langchain import AuscultCallbackHandler

handler = AuscultCallbackHandler(agent_type="triage-agent")
result = graph.invoke(inputs, config={"callbacks": [handler]})
print(handler.last_run_id)
```

### Codex CLI

`auscult codex` runs `codex exec --json` and records each completed item (agent messages, reasoning, shell commands, file changes, MCP tool calls, web searches) as a sanitized step:

```bash
uv run auscult codex "Summarize notes/visit.txt" --codex-arg=-s --codex-arg=read-only
```

To record a saved stream, use `codex exec --json "..." > events.jsonl` and then `uv run auscult codex --events events.jsonl`. In Python, use `run_codex` and `ingest_codex_events` from `auscult.integrations.codex`.

### Explicit capture

```python
from auscult import start_run, record_step

with start_run("triage-agent", initial_prompt=prompt):
    run_existing_agent(client, prompt)
    record_step("manual_annotation()", output="Review recommended")
```

The database URL is read from `DATABASE_URL`. SQLite is convenient for local demos; PostgreSQL is also supported.

## CLI

Every command accepts `--json` before the subcommand, as in `auscult --json runs`.

| Command | Purpose |
| --- | --- |
| `setup` | Download a spaCy model. |
| `migrate` | Apply database migrations. |
| `demo` | Capture one run with synthetic data. |
| `codex` | Run Codex (or ingest its `--json` output) as a captured run. |
| `sanitize` | Print files or stdin with PHI replaced; `--check` only reports. |
| `guard-hook` | Pre-tool hook that blocks agent reads of files with PHI. |
| `runs`, `run`, `steps` | List runs, or show one run and its steps. |
| `replay` | Play back a recorded run. |
| `compare` | Re-run a `module:function` handler against a run and diff the results. |
| `stats` | Failure rate, steps, latency, and redactions per agent type. |
| `export`, `purge` | Export a run as JSON or JSONL, or delete finished runs before a cutoff. |
| `eval` | Score PHI detection on the golden corpus. |

Use `uv run auscult <command> --help` for flags.
