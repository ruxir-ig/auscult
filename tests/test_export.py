"""Tests for export and retention helpers."""

from __future__ import annotations

import json
from datetime import timedelta

from auscult.capture import AuscultTracer
from auscult.db import get_session
from auscult.export import export_run, purge_runs_before
from auscult.models import Run, utcnow


def test_export_run_json_and_jsonl(db) -> None:
    tracer = AuscultTracer(agent_type="export-agent", initial_prompt="Check vitals.")
    tracer.record_step(llm_command="get_vitals()", output="BP 120/80.")
    tracer.finish()

    as_json = json.loads(export_run(tracer.run_id, fmt="json"))
    assert as_json["id"] == tracer.run_id
    assert as_json["agent_type"] == "export-agent"
    assert len(as_json["steps"]) == 1
    assert "entity_counts" in as_json

    lines = export_run(tracer.run_id, fmt="jsonl").strip().splitlines()
    assert json.loads(lines[0])["type"] == "run"
    assert json.loads(lines[1])["type"] == "step"


def test_purge_runs_before_dry_run_and_delete(db) -> None:
    old = AuscultTracer(agent_type="old-agent", initial_prompt="old")
    old.record_step(llm_command="x()", output="y")
    old.finish()

    fresh = AuscultTracer(agent_type="new-agent", initial_prompt="new")
    fresh.record_step(llm_command="x()", output="y")
    fresh.finish()

    with get_session() as session:
        run = session.get(Run, old.run_id)
        run.started_at = utcnow() - timedelta(days=30)
        session.commit()

    cutoff = utcnow() - timedelta(days=7)
    would = purge_runs_before(cutoff, dry_run=True)
    assert old.run_id in would
    assert fresh.run_id not in would

    deleted = purge_runs_before(cutoff, dry_run=False)
    assert old.run_id in deleted

    with get_session() as session:
        assert session.get(Run, old.run_id) is None
        assert session.get(Run, fresh.run_id) is not None
