"""CLI smoke tests for filtering, stats, compare, and --json."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

import pytest

from auscult import cli
from auscult.capture import AuscultTracer


def _seed_runs(db) -> tuple[str, str]:
    first = AuscultTracer(agent_type="triage-agent", initial_prompt="Check vitals.")
    first.record_step(llm_command="get_vitals()", output="BP 120/80.")
    first.finish()

    second = AuscultTracer(agent_type="billing-agent", initial_prompt="Invoice patient.")
    second.record_step(llm_command="create_invoice()", output=None, error_message="timeout")
    second.finish()
    return first.run_id, second.run_id


def test_cmd_runs_filters_and_json(db, capsys, monkeypatch) -> None:
    first_id, _ = _seed_runs(db)
    monkeypatch.setattr(
        sys,
        "argv",
        ["auscult", "--json", "runs", "--agent-type", "triage-agent", "--limit", "10"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["id"] == first_id
    assert payload[0]["agent_type"] == "triage-agent"


def test_cmd_runs_status_filter(db, capsys, monkeypatch) -> None:
    _seed_runs(db)
    monkeypatch.setattr(sys, "argv", ["auscult", "--json", "runs", "--status", "failed"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["status"] == "failed"
    assert payload[0]["agent_type"] == "billing-agent"


def test_cmd_stats_json(db, capsys, monkeypatch) -> None:
    _seed_runs(db)
    monkeypatch.setattr(sys, "argv", ["auscult", "--json", "stats"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    by_type = {row["agent_type"]: row for row in payload}
    assert by_type["triage-agent"]["runs"] == 1
    assert by_type["triage-agent"]["failed_runs"] == 0
    assert by_type["billing-agent"]["failed_runs"] == 1
    assert by_type["billing-agent"]["failure_rate"] == 1.0


def test_cmd_compare_matches(db, capsys, monkeypatch, tmp_path) -> None:
    run_id, _ = _seed_runs(db)
    handler_file = tmp_path / "match_handlers.py"
    handler_file.write_text(
        "def echo(cmd):\n"
        "    if cmd == 'get_vitals()':\n"
        "        return ('BP 120/80.', None)\n"
        "    return (None, 'timeout')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["auscult", "--json", "compare", run_id, "--handler", "match_handlers:echo"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_matched"] is True


def test_cmd_compare_exits_on_mismatch(db, monkeypatch, tmp_path) -> None:
    run_id, _ = _seed_runs(db)
    handler_file = tmp_path / "mismatch_handlers.py"
    handler_file.write_text("def bad(cmd):\n    return ('nope', None)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["auscult", "compare", run_id, "--handler", "mismatch_handlers:bad"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_cmd_runs_since_filter(db, capsys, monkeypatch) -> None:
    _seed_runs(db)
    future = (datetime.now() + timedelta(days=1)).isoformat()
    monkeypatch.setattr(sys, "argv", ["auscult", "--json", "runs", "--since", future])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload == []
