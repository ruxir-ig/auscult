"""Capture OpenAI Codex CLI sessions from the ``codex exec --json`` event stream.

Codex runs its own agent loop, so there is no client to wrap. Instead, Codex
emits one JSON event per line, and each completed item (agent message,
reasoning, shell command, file change, MCP tool call, web search, error)
becomes one sanitized step.

Run Codex under capture::

    from auscult.integrations.codex import run_codex

    result = run_codex("Summarize notes/visit.txt", codex_args=["-s", "read-only"])
    print(result.run_id, result.final_message)

Or ingest a saved stream (``codex exec --json "..." > events.jsonl``)::

    from auscult.integrations.codex import ingest_codex_events

    result = ingest_codex_events("events.jsonl", prompt="Summarize notes/visit.txt")

The CLI equivalents are ``auscult codex "<prompt>"`` and
``auscult codex --events events.jsonl``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ..capture import AuscultTracer
from ..db import get_session
from ..models import Step
from ..sanitizer import Sanitizer
from ._serialize import to_text


@dataclass
class CodexSummary:
    """What an event stream reported, independent of the stored steps."""

    thread_id: str | None = None
    final_message: str | None = None
    final_message_step: int | None = None
    failed: bool = False
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class CodexResult:
    run_id: str
    thread_id: str | None
    final_message: str | None
    failed: bool
    exit_code: int | None
    usage: dict[str, Any] | None


def record_codex_events(
    events: Iterable[str | Mapping[str, Any]],
    tracer: AuscultTracer,
    *,
    sanitizer: Sanitizer | None = None,
) -> CodexSummary:
    """Record each completed Codex item on ``tracer`` as a step.

    ``events`` are JSONL lines or already-decoded events. Blank lines and
    lines that do not start with ``{`` (CLI warnings) are skipped.
    ``item.started`` and ``item.updated`` events are ignored; only the final
    state of an item is recorded.

    The run is marked failed (with an error step) when a line starts with
    ``{`` but is not a JSON object, or when the stream ends before a turn
    completes, so a truncated capture is never reported as a success.

    ``summary.final_message`` is sanitized with ``sanitizer`` (default: a
    new :class:`Sanitizer` seeded from the run id), so its synthetic values
    can differ from the stored step's. For the exact stored text, read step
    ``summary.final_message_step`` after the tracer finishes, as
    :func:`run_codex` and :func:`ingest_codex_events` do. Do not pass a
    sanitizer that a background tracer's worker thread is using.
    """
    summary = CodexSummary()
    turn_open = False
    turns_finished = 0
    raw_final_message: str | None = None
    for line_number, raw in enumerate(events, start=1):
        try:
            event = _decode(raw)
        except ValueError:
            tracer.record_step(
                to_text({"type": "malformed_event"}),
                output=None,
                error_message=f"malformed Codex event on line {line_number}",
            )
            summary.failed = True
            continue
        if event is None:
            continue
        match event.get("type"):
            case "thread.started":
                summary.thread_id = event.get("thread_id")
            case "item.completed":
                item = event.get("item")
                if not isinstance(item, Mapping):
                    continue
                command, output, error = _item_step(item)
                step_index = tracer.step_count
                tracer.record_step(command, output=output, error_message=error)
                if item.get("type") == "agent_message":
                    raw_final_message = item.get("text")
                    summary.final_message_step = step_index
                if error is not None:
                    summary.failed = True
            case "turn.started":
                turn_open = True
            case "turn.completed":
                turn_open = False
                turns_finished += 1
                summary.usage = event.get("usage")
            case "turn.failed":
                turn_open = False
                turns_finished += 1
                failure = event.get("error")
                message = failure.get("message") if isinstance(failure, Mapping) else None
                tracer.record_step(
                    to_text({"type": "turn.failed"}),
                    output=None,
                    error_message=message or "Codex turn failed",
                )
                summary.failed = True
            case "error":
                tracer.record_step(
                    to_text({"type": "error"}),
                    output=None,
                    error_message=event.get("message") or "Codex stream error",
                )
                summary.failed = True
    if turn_open or turns_finished == 0:
        tracer.record_step(
            to_text({"type": "incomplete_stream"}),
            output=None,
            error_message="Codex stream ended before the turn completed",
        )
        summary.failed = True
    if raw_final_message:
        active = sanitizer or Sanitizer(seed=tracer.run_id)
        summary.final_message = active.sanitize(raw_final_message)
    return summary


def ingest_codex_events(
    path: str | Path,
    *,
    prompt: str | None = None,
    agent_type: str = "codex",
    **tracer_kwargs: Any,
) -> CodexResult:
    """Record a saved ``codex exec --json`` output file as one run."""
    path = Path(path)
    initial_prompt = prompt or f"codex exec events from {path.name}"
    with (
        AuscultTracer(
            agent_type=agent_type, initial_prompt=initial_prompt, **tracer_kwargs
        ) as tracer,
        path.open(encoding="utf-8") as fh,
    ):
        summary = record_codex_events(fh, tracer)
    return _result(tracer, summary, exit_code=None)


def run_codex(
    prompt: str,
    *,
    agent_type: str = "codex",
    codex_args: Sequence[str] = (),
    codex_bin: str = "codex",
    cwd: str | Path | None = None,
    **tracer_kwargs: Any,
) -> CodexResult:
    """Run ``codex exec --json`` and record its events as one run.

    ``codex_args`` are passed to ``codex exec`` before the prompt (e.g.
    ``["-s", "read-only", "-m", "gpt-5"]``). Codex's stderr is not captured.
    The run is marked failed if Codex reports an error or exits non-zero.
    """
    tracer = AuscultTracer(agent_type=agent_type, initial_prompt=prompt, **tracer_kwargs)
    try:
        process = subprocess.Popen(
            [codex_bin, "exec", "--json", *codex_args, "--", prompt],
            stdout=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        with process:
            summary = record_codex_events(process.stdout, tracer)
        exit_code = process.returncode
    except BaseException:
        tracer.finish(crashed=True)
        raise
    tracer.finish(crashed=exit_code != 0)
    return _result(tracer, summary, exit_code=exit_code)


def _result(tracer: AuscultTracer, summary: CodexSummary, *, exit_code: int | None) -> CodexResult:
    return CodexResult(
        run_id=tracer.run_id,
        thread_id=summary.thread_id,
        final_message=_stored_final_message(tracer, summary),
        failed=summary.failed or bool(exit_code),
        exit_code=exit_code,
        usage=summary.usage,
    )


def _stored_final_message(tracer: AuscultTracer, summary: CodexSummary) -> str | None:
    """The final agent message exactly as stored, so pseudonyms match the trace.

    Falls back to ``summary.final_message`` (also sanitized) when the run is
    not finished or the step is missing.
    """
    if not tracer.finished or summary.final_message_step is None:
        return summary.final_message
    with get_session() as session:
        stored = session.scalar(
            select(Step.output).where(
                Step.run_id == tracer.run_id,
                Step.step_index == summary.final_message_step,
            )
        )
    return stored if stored is not None else summary.final_message


def _decode(raw: str | Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the event, None for a line to skip, or raise ``ValueError``."""
    if isinstance(raw, Mapping):
        return raw
    line = raw.strip()
    if not line.startswith("{"):
        return None
    event = json.loads(line)  # JSONDecodeError is a ValueError
    if not isinstance(event, Mapping):
        raise ValueError("Codex event is not a JSON object")
    return event


def _item_step(item: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """Map a completed Codex item to ``(llm_command, output, error_message)``."""
    kind = item.get("type")
    status = item.get("status")
    failed = status == "failed"
    match kind:
        case "agent_message" | "reasoning":
            return to_text({"type": kind}), item.get("text"), None
        case "command_execution":
            exit_code = item.get("exit_code")
            error = None
            if failed or (isinstance(exit_code, int) and exit_code != 0):
                error = f"command exited with code {exit_code}"
            return (
                to_text({"type": kind, "command": item.get("command")}),
                item.get("aggregated_output"),
                error,
            )
        case "file_change":
            return (
                to_text({"type": kind, "changes": item.get("changes")}),
                status,
                "file change failed" if failed else None,
            )
        case "mcp_tool_call":
            error_payload = item.get("error")
            error = None
            if error_payload or failed:
                error = (
                    error_payload.get("message")
                    if isinstance(error_payload, Mapping)
                    else to_text(error_payload)
                ) or "MCP tool call failed"
            result = item.get("result")
            return (
                to_text(
                    {
                        "type": kind,
                        "server": item.get("server"),
                        "tool": item.get("tool"),
                        "arguments": item.get("arguments"),
                    }
                ),
                to_text(result) if result is not None else None,
                error,
            )
        case "web_search":
            return to_text({"type": kind, "query": item.get("query")}), None, None
        case "todo_list":
            return to_text({"type": kind}), to_text(item.get("items")), None
        case "error":
            return to_text({"type": kind}), None, item.get("message") or "Codex error"
        case _:
            return to_text({"type": kind}), to_text(dict(item)), None
