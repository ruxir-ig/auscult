# Auscult

Passive observability and safe replay tool for healthcare AI agents.

## Stack
- Python 3.12
- uv for package management (never use pip directly)
- SQLAlchemy 2.0 with mapped_column style (not the old Column style)
- PostgreSQL via psycopg2-binary
- Microsoft Presidio for PHI detection
- Faker for synthetic data generation

## Project structure
- src/auscult/ contains all package code
- models.py defines the Run and Step tables
- alembic/ contains database migrations
- db.py handles the database connection
- capture.py contains the AuscultTracer class
- replay.py loads and replays sanitized runs
- cli.py contains the command line interface

## Conventions
- Use Python 3.12 type hints everywhere
- Database URL is always read from the DATABASE_URL environment variable
- Never hardcode credentials
- Keep each file focused on one responsibility
- Run tasks with uv run, not python directly

## Running the project
- uv sync to install dependencies
- uv run migrate.py to apply Alembic migrations
- uv run alembic revision --autogenerate -m "message" to create schema revisions
- uv run auscult to use the CLI