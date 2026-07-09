import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    """Naive UTC timestamp for DateTime columns (replaces deprecated utcnow)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_agent_type_started_at", "agent_type", "started_at"),
        Index("ix_runs_status_started_at", "status", "started_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_type: Mapped[str] = mapped_column(String, nullable=False)
    initial_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, default="running")
    failed_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    redaction_count: Mapped[int] = mapped_column(Integer, default=0)
    # Entity type -> count only (no raw spans). Used for detection-quality audit.
    entity_counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    steps: Mapped[list["Step"]] = relationship("Step", back_populates="run")


class Step(Base):
    __tablename__ = "steps"
    __table_args__ = (
        UniqueConstraint("run_id", "step_index", name="uq_steps_run_id_step_index"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.id"), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_command: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    time_for_completion: Mapped[float | None] = mapped_column(nullable=True)
    redaction_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped["Run"] = relationship("Run", back_populates="steps")
