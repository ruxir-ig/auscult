# Auscult

Passive observability and safe replay for healthcare AI agents.

## Stack
- Python 3.12
- uv for package management (never use pip directly)
- SQLAlchemy 2.0 with mapped_column style (not the old Column style)
- PostgreSQL via psycopg2-binary (SQLite works for local demos and tests)
- Microsoft Presidio for PHI detection
- Faker for synthetic data generation

## Project structure
All package code lives in src/auscult/:
- models.py defines the Run and Step tables
- db.py handles the database connection
- sanitizer.py detects PHI and replaces it with synthetic values
- capture.py contains the AuscultTracer class
- writer.py runs the background write queue for `AuscultTracer(background=True)`
- context.py holds the ambient run context (`start_run`, `observe_run`, `record_step`)
- integrations/ wraps OpenAI and Anthropic clients, provides a LangChain callback handler, and captures Codex `exec --json` streams
- guard.py sanitizes files and tool results before an agent reads them (`auscult sanitize`, `auscult guard-hook`)
- demo.py captures one synthetic run for `auscult demo`
- replay.py loads and replays sanitized runs
- export.py exports runs as JSON/JSONL and purges old runs
- eval.py scores PHI detection against data/phi_eval_corpus.jsonl
- migrate.py applies the packaged Alembic migrations in alembic/
- cli.py contains the command line interface

## Conventions
- Use Python 3.12 type hints everywhere
- Database URL is always read from the DATABASE_URL environment variable
- Never hardcode credentials
- Keep each file focused on one responsibility
- Run tasks with uv run, not python directly

## Running the project
- uv sync --group nlp (or --group dev for tests) to install dependencies
- uv run auscult migrate to apply Alembic migrations
- uv run alembic revision --autogenerate -m "message" to create schema revisions
- uv run auscult to use the CLI
- AUSCULT_SPACY_MODEL=en_core_web_sm uv run pytest, uv run ruff check src tests, uv run mypy
