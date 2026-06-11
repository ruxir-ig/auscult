import os
import tempfile

# Must be set before auscult.db is imported anywhere.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.mkdtemp(prefix='auscult-tests-')}/test.sqlite"
)

import pytest

from auscult.db import engine
from auscult.models import Base


@pytest.fixture()
def db():
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
