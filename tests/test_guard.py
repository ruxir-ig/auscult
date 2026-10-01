"""Tests for sanitizing content before an agent reads it."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

import pytest

from auscult import cli
from auscult.guard import check_hook_payload, guard_tool, read_sanitized

PHI_TEXT = "Patient John Smith, phone 555-867-5309, reports chest pain.\n"
CLEAN_TEXT = "Recommend hydration and rest.\n"


@pytest.fixture()
def phi_file(tmp_path: Path) -> Path:
    path = tmp_path / "note.txt"
    path.write_text(PHI_TEXT)
    return path


@pytest.fixture()
def clean_file(tmp_path: Path) -> Path:
    path = tmp_path / "plan.txt"
    path.write_text(CLEAN_TEXT)
    return path


def test_read_sanitized_replaces_phi(phi_file: Path) -> None:
    text = read_sanitized(phi_file)
    assert "555-867-5309" not in text
    assert "chest pain" in text


def test_guard_tool_sanitizes_str_and_json_results() -> None:
    @guard_tool
    def fetch_note() -> str:
        return PHI_TEXT

    @guard_tool
    def fetch_record() -> dict:
        return {"phone": "555-867-5309", "visits": [1, 2]}

    assert "555-867-5309" not in fetch_note()
    record = fetch_record()
    assert record["phone"] != "555-867-5309"
    assert record["visits"] == [1, 2]


def test_guard_tool_async_and_rejects_unknown_types() -> None:
    @guard_tool()
    async def fetch_note() -> str:
        return PHI_TEXT

    @guard_tool
    def fetch_bytes() -> bytes:
        return PHI_TEXT.encode()

    assert "555-867-5309" not in asyncio.run(fetch_note())
    with pytest.raises(TypeError):
        fetch_bytes()


def test_hook_blocks_phi_and_allows_clean(phi_file: Path, clean_file: Path) -> None:
    blocked = check_hook_payload({"tool_name": "Read", "tool_input": {"file_path": str(phi_file)}})
    assert not blocked.allow
    assert blocked.reason is not None and "auscult sanitize" in blocked.reason
    assert "PHONE_NUMBER" in blocked.entity_counts

    assert check_hook_payload({"tool_input": {"path": str(clean_file)}}).allow
    assert check_hook_payload({"tool_input": {"command": "ls"}}).allow
    assert check_hook_payload({"tool_input": {"file_path": "/does/not/exist"}}).allow


def test_hook_fails_closed_on_unreadable_file(tmp_path: Path) -> None:
    binary = tmp_path / "scan.bin"
    binary.write_bytes(b"\xff\xfe\x00\x80")
    decision = check_hook_payload({"tool_input": {"file_path": str(binary)}})
    assert not decision.allow
    assert decision.reason is not None and "could not scan" in decision.reason


def test_cli_sanitize_and_check(phi_file: Path, clean_file: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["auscult", "sanitize", str(phi_file)])
    cli.main()
    out = capsys.readouterr().out
    assert "555-867-5309" not in out and "chest pain" in out

    monkeypatch.setattr(sys, "argv", ["auscult", "sanitize", "--check", str(clean_file)])
    cli.main()

    monkeypatch.setattr(sys, "argv", ["auscult", "--json", "sanitize", "--check", str(phi_file)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["redaction_count"] >= 1
    assert "text" not in payload[0]


def test_cli_sanitize_stdin(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(PHI_TEXT))
    monkeypatch.setattr(sys, "argv", ["auscult", "sanitize"])
    cli.main()
    assert "555-867-5309" not in capsys.readouterr().out


def test_cli_guard_hook_exit_codes(phi_file: Path, clean_file: Path, capsys, monkeypatch) -> None:
    def run_hook(path: Path) -> int:
        payload = {"tool_name": "Read", "tool_input": {"file_path": str(path)}}
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        monkeypatch.setattr(sys, "argv", ["auscult", "guard-hook"])
        try:
            cli.main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    assert run_hook(clean_file) == 0
    assert run_hook(phi_file) == 2
    assert "Blocked by auscult guard-hook" in capsys.readouterr().err
