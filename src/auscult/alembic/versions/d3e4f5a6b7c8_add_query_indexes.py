"""add query indexes and unique step constraint

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-07-09 10:50:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_runs_agent_type_started_at",
        "runs",
        ["agent_type", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_runs_status_started_at",
        "runs",
        ["status", "started_at"],
        unique=False,
    )
    # SQLite cannot ALTER ADD CONSTRAINT; use batch mode (no-op copy on Postgres).
    with op.batch_alter_table("steps") as batch_op:
        batch_op.create_unique_constraint(
            "uq_steps_run_id_step_index",
            ["run_id", "step_index"],
        )


def downgrade() -> None:
    with op.batch_alter_table("steps") as batch_op:
        batch_op.drop_constraint("uq_steps_run_id_step_index", type_="unique")
    op.drop_index("ix_runs_status_started_at", table_name="runs")
    op.drop_index("ix_runs_agent_type_started_at", table_name="runs")
