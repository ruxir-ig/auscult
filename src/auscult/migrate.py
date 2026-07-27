"""Apply packaged Alembic migrations.

End users run ``auscult migrate`` (or ``python -m auscult.migrate``) after
installing from PyPI / uv. Contributors can still use the repo-root
``migrate.py`` / ``alembic.ini`` wrappers.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def alembic_script_location() -> Path:
    """Return the directory that contains ``env.py`` and ``versions/``."""
    return Path(__file__).resolve().parent / "alembic"


def get_alembic_config() -> Config:
    """Build an Alembic config that points at the packaged migration scripts."""
    script_location = alembic_script_location()
    if not (script_location / "env.py").is_file():
        raise FileNotFoundError(
            f"Packaged Alembic scripts not found at {script_location}. "
            "Reinstall auscult or run migrations from a source checkout."
        )
    cfg = Config()
    cfg.set_main_option("script_location", str(script_location))
    return cfg


def upgrade_head() -> None:
    """Apply all pending migrations (``alembic upgrade head``)."""
    command.upgrade(get_alembic_config(), "head")


def main() -> None:
    upgrade_head()
    print("Migrations applied.")


if __name__ == "__main__":
    main()
