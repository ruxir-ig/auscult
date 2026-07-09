import tempfile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from auscult.db import get_engine, reset_connection_state


@pytest.fixture()
def migration_db(monkeypatch: pytest.MonkeyPatch) -> None:
    db_dir = tempfile.mkdtemp(prefix="auscult-migrations-")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_dir}/test.sqlite")
    reset_connection_state()


def test_migrations_apply_from_empty_db(migration_db: None) -> None:
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")

    inspector = inspect(get_engine())
    tables = set(inspector.get_table_names())

    assert "runs" in tables
    assert "steps" in tables
    assert "alembic_version" in tables

    run_indexes = {idx["name"] for idx in inspector.get_indexes("runs")}
    assert "ix_runs_agent_type_started_at" in run_indexes
    assert "ix_runs_status_started_at" in run_indexes

    step_uniques = {
        tuple(uq["column_names"])
        for uq in inspector.get_unique_constraints("steps")
    }
    assert ("run_id", "step_index") in step_uniques


def test_migrations_are_reversible(migration_db: None) -> None:
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    inspector = inspect(get_engine())
    assert "runs" not in inspector.get_table_names()
    assert "steps" not in inspector.get_table_names()
