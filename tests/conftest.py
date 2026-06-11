import os
import tempfile

# Must be set before the engine is first created.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.mkdtemp(prefix='auscult-tests-')}/test.sqlite"
)

import pytest

from auscult.db import get_engine
from auscult.models import Base


@pytest.fixture()
def db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
