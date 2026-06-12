import os
import tempfile

# Must be set before the engine is first created.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.mkdtemp(prefix='auscult-tests-')}/test.sqlite"
)

import pytest
from alembic import command
from alembic.config import Config
from auscult.db import get_engine, reset_connection_state
from auscult.models import Base


@pytest.fixture()
def db():
    reset_connection_state()
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    reset_connection_state()


@pytest.fixture()
def migrated_db(monkeypatch: pytest.MonkeyPatch):
    """Apply Alembic migrations on an isolated SQLite database."""
    db_dir = tempfile.mkdtemp(prefix="auscult-integration-")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_dir}/test.sqlite")
    reset_connection_state()
    command.upgrade(Config("alembic.ini"), "head")
    yield
    reset_connection_state()
