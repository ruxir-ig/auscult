"""Tests for capturing Codex `exec --json` event streams."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from auscult import cli
from auscult.capture import AuscultTracer
from auscult.db import get_session
from auscult.integrations.codex import ingest_codex_events, record_codex_events, run_codex
from auscult.models import Run, Step

FIXTURE = Path(__file__).parent / "fixtures" / "codex_exec_events.jsonl"


def _load(run_id: str) -> tuple[Run, list[Step]]:
    with get_session() as session:
        run = session.get(Run, run_id)
        assert run is not None
        steps = list(
            session.scalars(select(Step).where(Step.run_id == run_id).order_by(Step.step_index))
        )
        session.expunge_all()
        return run, steps


def test_ingest_recorded_stream(db) -> None:
    result = ingest_codex_events(FIXTURE, prompt="Read the note")

    run, steps = _load(result.run_id)
    assert run.agent_type == "codex"
    assert run.status == "completed"
    assert result.thread_id == "01a0f774-9483-7e91-844b-c4d72f038782"
    assert result.final_message == "done"
    assert result.usage is not None and result.usage["output_tokens"] == 49
    # item.started events are skipped; three completed items remain.
    assert [json.loads(s.llm_command)["type"] for s in steps] == [
        "agent_message",
        "command_execution",
        "agent_message",
    ]
    assert json.loads(steps[1].llm_command)["command"] == "/bin/zsh -lc 'cat note.txt'"
    assert (steps[1].output or "").startswith("Patient note: follow up in")


def test_failures_mark_run_failed(db) -> None:
    events = [
        "not json: a CLI warning",
        "",
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "false",
                "aggregated_output": "",
                "exit_code": 1,
                "status": "failed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "ehr",
                "tool": "lookup",
                "arguments": {"id": 1},
                "result": None,
                "error": {"message": "unauthorized"},
                "status": "failed",
            },
        },
        {"type": "turn.failed", "error": {"message": "stream disconnected"}},
    ]
    with AuscultTracer(agent_type="codex", initial_prompt="x") as tracer:
        summary = record_codex_events(events, tracer)

    run, steps = _load(tracer.run_id)
    assert summary.failed
    assert run.status == "failed"
    assert run.failed_step == 0
    assert [s.error_message for s in steps] == [
        "command exited with code 1",
        "unauthorized",
        "stream disconnected",
    ]


def test_phi_in_codex_output_is_sanitized(db) -> None:
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": "Patient John Smith can be reached at 555-867-5309.",
            },
        }
    ]
    with AuscultTracer(agent_type="codex", initial_prompt="x") as tracer:
        record_codex_events(events, tracer)

    _, steps = _load(tracer.run_id)
    assert "555-867-5309" not in (steps[0].output or "")
    assert steps[0].redaction_count >= 1


@pytest.fixture()
def fake_codex(tmp_path: Path) -> Path:
    """An executable that replays the fixture like `codex exec --json` would."""
    script = tmp_path / "codex"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"assert sys.argv[1:3] == ['exec', '--json'], sys.argv\n"
        f"sys.stdout.write(open({str(FIXTURE)!r}).read())\n"
        "sys.exit(int(sys.argv[-1] == 'fail'))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_run_codex_subprocess(db, fake_codex: Path) -> None:
    result = run_codex("Read the note", codex_bin=str(fake_codex), codex_args=["-s", "read-only"])
    run, steps = _load(result.run_id)
    assert result.exit_code == 0
    assert not result.failed
    assert run.status == "completed"
    assert run.initial_prompt == "Read the note"
    assert len(steps) == 3


def test_run_codex_nonzero_exit_fails_run(db, fake_codex: Path) -> None:
    result = run_codex("fail", codex_bin=str(fake_codex))
    run, _ = _load(result.run_id)
    assert result.exit_code == 1
    assert result.failed
    assert run.status == "failed"


def test_cli_codex_events_json(db, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["auscult", "--json", "codex", "--events", str(FIXTURE), "Read the note"]
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["final_message"] == "done"
    assert payload["failed"] is False
    _, steps = _load(payload["run_id"])
    assert len(steps) == 3
