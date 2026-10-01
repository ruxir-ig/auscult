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

from ..capture import AuscultTracer
from ._serialize import to_text


@dataclass
class CodexSummary:
    """What an event stream reported, independent of the stored steps."""

    thread_id: str | None = None
    final_message: str | None = None
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
    events: Iterable[str | Mapping[str, Any]], tracer: AuscultTracer
) -> CodexSummary:
    """Record each completed Codex item on ``tracer`` as a step.

    ``events`` are JSONL lines or already-decoded events. Lines that are not
    JSON objects (blank lines, CLI warnings) are skipped. ``item.started`` and
    ``item.updated`` events are ignored; only the final state of an item is
    recorded.
    """
    summary = CodexSummary()
    for raw in events:
        event = _decode(raw)
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
                tracer.record_step(command, output=output, error_message=error)
                if item.get("type") == "agent_message":
                    summary.final_message = item.get("text")
                if error is not None:
                    summary.failed = True
            case "turn.completed":
                summary.usage = event.get("usage")
            case "turn.failed":
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
        final_message=summary.final_message,
        failed=summary.failed or bool(exit_code),
        exit_code=exit_code,
        usage=summary.usage,
    )


def _decode(raw: str | Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        return raw
    line = raw.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, Mapping) else None


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
