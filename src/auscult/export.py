"""Export and retention helpers for sanitized runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

from .db import get_session
from .models import Run, Step
from .replay import fetch_run


def run_to_dict(run: Run) -> dict[str, Any]:
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
    }


def step_to_dict(step: Step) -> dict[str, Any]:
    return {
        "id": step.id,
        "run_id": step.run_id,
        "step_index": step.step_index,
        "llm_command": step.llm_command,
        "output": step.output,
        "error_message": step.error_message,
        "time_for_completion": step.time_for_completion,
        "redaction_count": step.redaction_count,
        "entity_counts": step.entity_counts or {},
    }


def export_run(run_id: str, *, fmt: str = "json") -> str:
    """Return a sanitized run as JSON or JSONL text."""
    with get_session() as session:
        run, steps = fetch_run(session, run_id)
        run_dict = run_to_dict(run)
        step_dicts = [step_to_dict(step) for step in steps]

    if fmt == "json":
        return json.dumps({**run_dict, "steps": step_dicts}, indent=2, default=str) + "\n"
    if fmt == "jsonl":
        records = [{"type": "run", **run_dict}, *({"type": "step", **s} for s in step_dicts)]
        return "".join(json.dumps(record) + "\n" for record in records)
    raise ValueError(f"unsupported export format: {fmt!r} (use json or jsonl)")


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
        query = select(Run.id).where(
            Run.started_at < before,
            Run.status != "running",
        )
        if agent_type is not None:
            query = query.where(Run.agent_type == agent_type)
        run_ids = list(session.scalars(query))
        if dry_run or not run_ids:
            return run_ids

        session.execute(delete(Step).where(Step.run_id.in_(run_ids)))
        session.execute(delete(Run).where(Run.id.in_(run_ids)))
        session.commit()
        return run_ids
