"""add redaction_count to runs and steps

Revision ID: b1c2d3e4f5a6
Revises: 9085ad27b4c5
Create Date: 2026-07-09 08:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "9085ad27b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("redaction_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "steps",
        sa.Column("redaction_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("steps", "redaction_count")
    op.drop_column("runs", "redaction_count")
