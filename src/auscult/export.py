"""Export and retention helpers for sanitized runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, TextIO

from sqlalchemy import delete, select

from .db import get_session
from .models import Run, Step
from .replay import RunNotFoundError


def run_to_export_dict(run: Run, steps: list[Step]) -> dict[str, Any]:
    return {
        "id": run.id,
        "agent_type": run.agent_type,
        "status": run.status,
        "total_steps": run.total_steps,
        "failed_step": run.failed_step,
        "redaction_count": run.redaction_count,
        "entity_counts": run.entity_counts or {},
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "initial_prompt": run.initial_prompt,
        "steps": [
            {
                "id": step.id,
                "step_index": step.step_index,
                "llm_command": step.llm_command,
                "output": step.output,
                "error_message": step.error_message,
                "time_for_completion": step.time_for_completion,
                "redaction_count": step.redaction_count,
                "entity_counts": step.entity_counts or {},
            }
            for step in steps
        ],
    }


def export_run(run_id: str, *, fmt: str = "json") -> str:
    """Return a sanitized run as JSON or JSONL text."""
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        steps = list(
            session.execute(
                select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
            )
            .scalars()
            .all()
        )
        payload = run_to_export_dict(run, steps)

    if fmt == "json":
        return json.dumps(payload, indent=2, default=str) + "\n"
    if fmt == "jsonl":
        lines = [json.dumps({"type": "run", **{k: v for k, v in payload.items() if k != "steps"}})]
        for step in payload["steps"]:
            lines.append(json.dumps({"type": "step", "run_id": run_id, **step}))
        return "\n".join(lines) + "\n"
    raise ValueError(f"unsupported export format: {fmt!r} (use json or jsonl)")


def write_export(run_id: str, dest: TextIO, *, fmt: str = "json") -> None:
    dest.write(export_run(run_id, fmt=fmt))


def purge_runs_before(
    before: datetime,
    *,
    dry_run: bool = False,
    agent_type: str | None = None,
) -> list[str]:
    """Delete finished runs (and their steps) started before ``before``.

    Returns the list of deleted (or would-be-deleted) run ids. Runs still
    ``running`` are never purged.
    """
    with get_session() as session:
        query = select(Run).where(
            Run.started_at < before,
            Run.status != "running",
        )
        if agent_type is not None:
            query = query.where(Run.agent_type == agent_type)
        runs = list(session.execute(query).scalars().all())
        run_ids = [run.id for run in runs]
        if dry_run or not run_ids:
            return run_ids

        session.execute(delete(Step).where(Step.run_id.in_(run_ids)))
        session.execute(delete(Run).where(Run.id.in_(run_ids)))
        session.commit()
        return run_ids
