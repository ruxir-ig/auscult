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
export DATABASE_URL=sqlite:////tmp/auscult-demo.sqlite
uv run auscult migrate
```

Create a run with synthetic example data:

```python
from auscult import AuscultTracer

with AuscultTracer(
    agent_type="triage-agent",
    initial_prompt="A patient reports a rash. Classify the symptoms.",
) as tracer:
    tracer.record_step(
        llm_command="Classify the patient's symptoms.",
        output="The patient reports a rash. Recommend clinical review.",
    )

print(tracer.run_id)
```

Save this as `demo.py` and run `uv run python demo.py`. Then inspect and replay the run:

```bash
uv run auscult run <RUN_ID>
uv run auscult replay <RUN_ID>
```

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

Anthropic client wrapping and LangChain/LangGraph callbacks are also supported. Install the optional LangChain dependency with `uv add 'auscult[langchain]'`. See [the integration examples](#integration-options) below.

## What Auscult captures

- Runs and ordered steps, including prompts, model commands, outputs, and errors.
- PHI detection and synthetic replacements before trace data is written.
- Redaction totals and counts by detected entity type.
- CLI tools to inspect, replay, compare, export, and summarize runs.

Capture can also be used explicitly with `AuscultTracer`, `start_run`, and `record_step`. See [the API examples](#integration-options).

## What I’d improve next

1. **Add support for the Codex harness API.** Make Auscult easy to start with and test through the harness developers can access most readily.
2. **Prevent PHI from being consumed in the first place.** Add protection at the point agents read files, so sensitive content is detected or masked before it enters the agent context.
3. **Make Auscult easy to try and improve.** Provide an accessible testing experience and gather feedback from health professionals to guide improvements.

## Important limitation

PHI protection is detection-based. Detectors can miss sensitive information, so validate performance on representative data and treat stored traces as sensitive until your organization has approved their use. Do not use real patient data for a demo. Development is ongoing, in order to mitigate this.

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

### Explicit capture

```python
from auscult import start_run, record_step

with start_run("triage-agent", initial_prompt=prompt):
    run_existing_agent(client, prompt)
    record_step("manual_annotation()", output="Review recommended")
```

The database URL is read from `DATABASE_URL`. SQLite is convenient for local demos; PostgreSQL is also supported. Use `uv run auscult --help` to see CLI commands.
