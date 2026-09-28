# Auscult roadmap

Prioritized follow-ups. Items marked **done** shipped; the rest are next.

## Done

- [x] Harden sanitizer: configurable spaCy model, `score_threshold`, clinical
      allow-list, org deny-list, JSON leaf sanitization, `redaction_count`
- [x] Fail-closed capture on sanitizer errors; production TLS / at-rest guidance
- [x] CLI: `compare`, `runs` filters, `stats`, `--json`
- [x] CI + ruff + mypy
- [x] Deterministic Faker seeding from `run_id`
- [x] Structured entity audit (`entity_counts` type→count, no raw spans)
- [x] Background writer (`AuscultTracer(..., background=True)`)
- [x] `auscult export` / `auscult purge` (retention)
- [x] Golden-set PHI eval harness (`auscult eval`, packaged corpus)
- [x] Optional dual-pass / ensemble NER (`AUSCULT_DUAL_PASS_MODEL`)
- [x] Optional spaCy model packaging (`nlp` / `nlp-trf` dependency groups; `sm` in dev group)
- [x] DB indexes: `runs(agent_type, started_at)`, `runs(status, started_at)`,
      unique `(run_id, step_index)` on `steps`
- [x] Pluggable capture: ambient run context (`start_run` / `@observe_run`),
      OpenAI/Anthropic client wrappers, LangChain/LangGraph callback handler
- [x] PyPI-ready packaging: MIT license, metadata, `auscult setup` / `auscult migrate`,
      spaCy models via post-install (not direct URL deps), Trusted Publishing workflow

## Later / ideas

- More framework adapters: CrewAI, AutoGen, LlamaIndex, OpenAI Agents SDK
- OpenTelemetry span exporter so Auscult runs appear in existing APM traces
- Sample-and-audit UI for false-negative review
- Per-tenant allow/deny list config files
- Metrics export (Prometheus) for redaction rate / queue depth
- Expand eval corpus with site-specific PHI patterns
- Wire `auscult eval` into CI as a regression gate with a minimum F1 threshold
