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

## Next

### Golden-set PHI eval harness
`redaction_count` / `entity_counts` monitor volume and mix, not quality. Add a
checked-in corpus of labeled clinical snippets (true PHI vs eponyms / drug
names / lab values) and `uv run auscult eval` reporting precision/recall.
That is how we know `lg` → `trf` or threshold tweaks actually help.

### Optional dual-pass / ensemble NER
Always run pattern recognizers; optionally add a second NER pass
(`en_core_web_trf` or a medical NER) only on PERSON/LOCATION candidates.
Better recall without paying transformer cost on every token.

### Optional small spaCy model packaging
Make `en_core_web_lg` the only default dependency; put `en_core_web_sm` behind
a uv/pip extra so production installs are not forced to carry both wheels.

### DB indexes (when volume grows)
Add indexes so filtered list/stats queries stay cheap:

- `runs(agent_type, started_at)`
- `runs(status, started_at)`
- unique `(run_id, step_index)` on `steps`

Not urgent at low volume; add via Alembic once `auscult runs --since` or
`stats` starts scanning large tables. See README note under "Scale".

## Later / ideas

- Sample-and-audit UI for false-negative review
- Per-tenant allow/deny list config files
- Metrics export (Prometheus) for redaction rate / queue depth
