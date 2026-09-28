"""Database access, initialized lazily.

DATABASE_URL is only read when the engine is first needed, so importing
the package (e.g. for `auscult --help`) works without any DB configured.
"""

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(os.environ["DATABASE_URL"])


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False)


def get_session() -> Session:
    return _session_factory()()


def reset_connection_state() -> None:
    """Clear cached engine and session factory (for tests)."""
    get_engine.cache_clear()
    _session_factory.cache_clear()
