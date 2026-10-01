"""Drop-in integrations for existing agent stacks.

These adapters let host applications adopt Auscult without rewriting their
agent loops:

- :mod:`auscult.integrations.openai` — wrap an OpenAI (or OpenAI-compatible)
  client so every completion call is captured as a step.
- :mod:`auscult.integrations.anthropic` — same for Anthropic clients.
- :mod:`auscult.integrations.langchain` — a LangChain / LangGraph callback
  handler; pass it via ``config={"callbacks": [...]}``.
- :mod:`auscult.integrations.codex` — record a Codex CLI session from its
  ``codex exec --json`` event stream (live or saved).

All adapters resolve the tracer from the ambient run context
(:mod:`auscult.context`) unless one is passed explicitly, so a single
``start_run`` / ``@observe_run`` at the agent entry point is enough.
"""
