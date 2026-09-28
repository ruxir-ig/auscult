"""Replay recorded agent runs for debugging, QA, and audit.

Runs are loaded from the database as already-sanitized traces. Replay never
re-exposes raw PHI — it only reads what capture stored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_session
from .models import Run, Step


class RunNotFoundError(LookupError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} not found")
        self.run_id = run_id


@dataclass(frozen=True)
class ReplayStep:
    step_index: int
    llm_command: str
    output: str | None
    error_message: str | None
    time_for_completion: float | None


@dataclass(frozen=True)
class ReplayRun:
    run_id: str
    agent_type: str
    initial_prompt: str
    status: str
    failed_step: int | None
    steps: tuple[ReplayStep, ...]


@dataclass(frozen=True)
class StepComparison:
    step_index: int
    llm_command: str
    expected_output: str | None
    actual_output: str | None
    expected_error: str | None
    actual_error: str | None

    @property
    def output_match(self) -> bool:
        return self.expected_output == self.actual_output

    @property
    def error_match(self) -> bool:
        return self.expected_error == self.actual_error

    @property
    def matched(self) -> bool:
        return self.output_match and self.error_match


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    steps: tuple[StepComparison, ...]

    @property
    def all_matched(self) -> bool:
        return all(step.matched for step in self.steps)


def fetch_run(session: Session, run_id: str) -> tuple[Run, list[Step]]:
    """Return a run and its steps in order, or raise :class:`RunNotFoundError`."""
    run = session.get(Run, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    steps = session.scalars(
        select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
    )
    return run, list(steps)


def load_run(run_id: str) -> ReplayRun:
    with get_session() as session:
        run, steps = fetch_run(session, run_id)
        return ReplayRun(
            run_id=run.id,
            agent_type=run.agent_type,
            initial_prompt=run.initial_prompt,
            status=run.status,
            failed_step=run.failed_step,
            steps=tuple(
                ReplayStep(
                    step_index=step.step_index,
                    llm_command=step.llm_command,
                    output=step.output,
                    error_message=step.error_message,
                    time_for_completion=step.time_for_completion,
                )
                for step in steps
            ),
        )


class RunReplayer:
    """Replays a sanitized run step by step."""

    def __init__(self, run: ReplayRun) -> None:
        self.run = run

    @classmethod
    def from_run_id(cls, run_id: str) -> RunReplayer:
        return cls(load_run(run_id))

    def playback(self) -> Iterator[ReplayStep]:
        """Yield recorded steps in order without invoking an agent."""
        yield from self.run.steps

    def replay(
        self,
        handler: Callable[[str], tuple[str | None, str | None]],
    ) -> ReplayResult:
        """Invoke handler for each step and compare results to the recording."""
        comparisons = []
        for step in self.run.steps:
            actual_output, actual_error = handler(step.llm_command)
            comparisons.append(
                StepComparison(
                    step_index=step.step_index,
                    llm_command=step.llm_command,
                    expected_output=step.output,
                    actual_output=actual_output,
                    expected_error=step.error_message,
                    actual_error=actual_error,
                )
            )
        return ReplayResult(run_id=self.run.run_id, steps=tuple(comparisons))


def format_playback(run: ReplayRun) -> str:
    """Format a run as human-readable playback output."""
    lines = [
        f"Replaying run {run.run_id} ({run.agent_type}, {run.status})",
        f"initial_prompt: {run.initial_prompt}",
        "",
    ]
    for step in run.steps:
        lines.append(f"step {step.step_index}: {step.llm_command}")
        if step.error_message is not None:
            lines.append(f"  ! {step.error_message}")
        elif step.output is not None:
            lines.append(f"  -> {step.output}")
        else:
            lines.append("  -> —")
        if step.time_for_completion is not None:
            lines.append(f"  ({step.time_for_completion:.3f}s recorded)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
