"""Command-line interface for inspecting and replaying sanitized runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any, NoReturn

from sqlalchemy import select

from .db import get_session
from .demo import DEFAULT_DEMO_DATABASE_URL
from .eval import evaluate, format_report
from .export import export_run, purge_runs_before, run_to_dict, step_to_dict
from .migrate import upgrade_head
from .models import Run, Step
from .replay import RunNotFoundError, RunReplayer, fetch_run, format_playback
from .sanitizer import DEFAULT_SPACY_MODEL, Sanitizer, SanitizeResult


def _print_run(run: Run) -> None:
    finished = run.finished_at.isoformat() if run.finished_at else "—"
    failed_step = run.failed_step if run.failed_step is not None else "—"
    print(f"run {run.id}")
    print(f"  agent_type:      {run.agent_type}")
    print(f"  status:          {run.status}")
    print(f"  total_steps:     {run.total_steps}")
    print(f"  failed_step:     {failed_step}")
    print(f"  redaction_count: {run.redaction_count}")
    print(f"  entity_counts:   {run.entity_counts or {}}")
    print(f"  started_at:      {run.started_at.isoformat()}")
    print(f"  finished_at:     {finished}")
    print(f"  initial_prompt:  {run.initial_prompt}")


def _print_step(step: Step) -> None:
    print(f"step {step.step_index} ({step.id})")
    print(f"  llm_command:     {step.llm_command}")
    print(f"  output:          {step.output if step.output is not None else '—'}")
    if step.error_message is not None:
        print(f"  error:           {step.error_message}")
    if step.time_for_completion is not None:
        print(f"  duration:        {step.time_for_completion:.3f}s")
    print(f"  redaction_count: {step.redaction_count}")
    print(f"  entity_counts:   {step.entity_counts or {}}")


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _exit_not_found(exc: RunNotFoundError) -> NoReturn:
    print(str(exc), file=sys.stderr)
    sys.exit(1)


def _parse_since(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (date or datetime)."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --since value {value!r}; "
            "use ISO-8601 (e.g. 2026-07-01 or 2026-07-01T12:00:00)"
        ) from exc


def _load_handler(spec: str) -> Callable[[str], tuple[str | None, str | None]]:
    """Load ``module:function`` and return the callable."""
    if ":" not in spec:
        raise ValueError("handler must be in module:function form")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    handler = getattr(module, func_name)
    if not callable(handler):
        raise TypeError(f"{spec} is not callable")
    return handler  # type: ignore[no-any-return]


def _cmd_runs(
    *,
    agent_type: str | None,
    status: str | None,
    since: datetime | None,
    limit: int | None,
    as_json: bool,
) -> None:
    with get_session() as session:
        query = select(Run).order_by(Run.started_at.desc())
        if agent_type is not None:
            query = query.where(Run.agent_type == agent_type)
        if status is not None:
            query = query.where(Run.status == status)
        if since is not None:
            query = query.where(Run.started_at >= since)
        if limit is not None:
            query = query.limit(limit)

        runs = session.scalars(query).all()

        if as_json:
            _emit_json([run_to_dict(run) for run in runs])
            return

        if not runs:
            print("No runs found.")
            return

        for run in runs:
            _print_run(run)
            print()


def _cmd_run(run_id: str, *, as_json: bool) -> None:
    with get_session() as session:
        try:
            run, steps = fetch_run(session, run_id)
        except RunNotFoundError as exc:
            _exit_not_found(exc)

        if as_json:
            _emit_json({**run_to_dict(run), "steps": [step_to_dict(step) for step in steps]})
            return

        _print_run(run)
        print(f"\n{len(steps)} step(s):")
        for step in steps:
            print()
            _print_step(step)


def _cmd_steps(run_id: str, *, as_json: bool) -> None:
    with get_session() as session:
        try:
            _, steps = fetch_run(session, run_id)
        except RunNotFoundError as exc:
            _exit_not_found(exc)

        if as_json:
            _emit_json([step_to_dict(step) for step in steps])
            return

        if not steps:
            print(f"No steps recorded for run {run_id}.")
            return

        for step in steps:
            _print_step(step)
            print()


def _cmd_replay(run_id: str, *, as_json: bool) -> None:
    try:
        run = RunReplayer.from_run_id(run_id).run
    except RunNotFoundError as exc:
        _exit_not_found(exc)

    if as_json:
        _emit_json(asdict(run))
        return

    print(format_playback(run), end="")


def _cmd_compare(run_id: str, handler_spec: str, *, as_json: bool) -> None:
    try:
        handler = _load_handler(handler_spec)
    except Exception as exc:
        print(f"Failed to load handler {handler_spec!r}: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        result = RunReplayer.from_run_id(run_id).replay(handler)
    except RunNotFoundError as exc:
        _exit_not_found(exc)

    if as_json:
        _emit_json(
            {
                "run_id": result.run_id,
                "all_matched": result.all_matched,
                "steps": [
                    {
                        **asdict(step),
                        "output_match": step.output_match,
                        "error_match": step.error_match,
                        "matched": step.matched,
                    }
                    for step in result.steps
                ],
            }
        )
    else:
        matched = sum(1 for step in result.steps if step.matched)
        print(
            f"Compare run {result.run_id}: "
            f"{matched}/{len(result.steps)} steps matched"
            f"{' (all matched)' if result.all_matched else ''}"
        )
        for step in result.steps:
            mark = "OK" if step.matched else "DIFF"
            print(f"  [{mark}] step {step.step_index}: {step.llm_command}")
            if not step.output_match:
                print(f"       expected output: {step.expected_output!r}")
                print(f"       actual output:   {step.actual_output!r}")
            if not step.error_match:
                print(f"       expected error:  {step.expected_error!r}")
                print(f"       actual error:    {step.actual_error!r}")

    if not result.all_matched:
        sys.exit(1)


def _cmd_stats(*, agent_type: str | None, as_json: bool) -> None:
    with get_session() as session:
        runs_query = select(Run)
        if agent_type is not None:
            runs_query = runs_query.where(Run.agent_type == agent_type)
        runs = session.scalars(runs_query).all()

        runs_by_type: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            runs_by_type[run.agent_type].append(run)

        run_agent = {run.id: run.agent_type for run in runs}
        latencies: dict[str, list[float]] = defaultdict(list)
        if run_agent:
            steps = session.scalars(select(Step).where(Step.run_id.in_(list(run_agent))))
            for step in steps:
                if step.time_for_completion is not None:
                    latencies[run_agent[step.run_id]].append(step.time_for_completion)

    rows: list[dict[str, Any]] = []
    for agent, agent_runs in sorted(runs_by_type.items()):
        run_count = len(agent_runs)
        failed_runs = sum(1 for run in agent_runs if run.status == "failed")
        agent_latencies = latencies[agent]
        rows.append(
            {
                "agent_type": agent,
                "runs": run_count,
                "failed_runs": failed_runs,
                "failure_rate": failed_runs / run_count,
                "avg_steps": sum(run.total_steps for run in agent_runs) / run_count,
                "avg_step_latency": (
                    sum(agent_latencies) / len(agent_latencies) if agent_latencies else None
                ),
                "avg_redactions": sum(run.redaction_count for run in agent_runs) / run_count,
            }
        )

    if as_json:
        _emit_json(rows)
        return

    if not rows:
        print("No runs found.")
        return

    print(
        f"{'agent_type':<24} {'runs':>6} {'fail%':>7} {'avg_steps':>10} "
        f"{'avg_lat':>10} {'avg_redact':>10}"
    )
    for row in rows:
        fail_pct = f"{row['failure_rate'] * 100:.1f}%"
        avg_lat = (
            f"{row['avg_step_latency']:.3f}s"
            if row["avg_step_latency"] is not None
            else "—"
        )
        print(
            f"{row['agent_type']:<24} {row['runs']:>6} {fail_pct:>7} "
            f"{row['avg_steps']:>10.1f} {avg_lat:>10} {row['avg_redactions']:>10.1f}"
        )


def _cmd_export(run_id: str, *, fmt: str, output: str | None) -> None:
    try:
        text = export_run(run_id, fmt=fmt)
    except RunNotFoundError as exc:
        _exit_not_found(exc)

    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {fmt} export of run {run_id} to {output}")
    else:
        sys.stdout.write(text)


def _cmd_purge(
    *,
    before: datetime,
    dry_run: bool,
    agent_type: str | None,
    as_json: bool,
) -> None:
    deleted = purge_runs_before(before, dry_run=dry_run, agent_type=agent_type)
    if as_json:
        _emit_json(
            {
                "dry_run": dry_run,
                "before": before.isoformat(),
                "count": len(deleted),
                "run_ids": deleted,
            }
        )
        return
    verb = "Would delete" if dry_run else "Deleted"
    print(f"{verb} {len(deleted)} run(s) started before {before.isoformat()}.")
    for run_id in deleted:
        print(f"  {run_id}")


def _cmd_migrate(*, as_json: bool) -> None:
    upgrade_head()
    if as_json:
        _emit_json({"status": "ok", "message": "Migrations applied."})
    else:
        print("Migrations applied.")


def _cmd_setup(*, model: str, as_json: bool) -> None:
    """Download a spaCy model (models are not published on PyPI)."""
    from spacy.cli.download import download as spacy_download

    spacy_download(model)
    if as_json:
        _emit_json({"status": "ok", "model": model})
    else:
        print(f"Installed spaCy model {model!r}.")


def _cmd_eval(
    *,
    corpus: str | None,
    nlp_model: str | None,
    dual_pass_model: str | None,
    score_threshold: float | None,
    verbose: bool,
    as_json: bool,
) -> None:
    sanitizer = Sanitizer(
        nlp_model=nlp_model,
        dual_pass_model=dual_pass_model,
        score_threshold=score_threshold,
    )
    report = evaluate(sanitizer=sanitizer, corpus_path=corpus)
    if as_json:
        _emit_json(report.to_dict())
    else:
        print(format_report(report, verbose=verbose), end="")
    # Non-zero exit when recall is catastrophically low (likely misconfig).
    if report.recall < 0.5 and report.true_positives + report.false_negatives > 0:
        sys.exit(1)


def _cmd_sanitize(paths: list[str], *, check: bool, as_json: bool) -> None:
    """Print sanitized file contents (or stdin) so agents read masked text."""
    sanitizer = Sanitizer()
    sources: list[tuple[str, SanitizeResult]] = []
    if not paths:
        sources.append(("-", sanitizer.sanitize_with_stats(sys.stdin.read())))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                sources.append((path, sanitizer.sanitize_with_stats(fh.read())))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Cannot read {path}: {exc}", file=sys.stderr)
            sys.exit(2)

    found = any(result.redaction_count for _, result in sources)
    if as_json:
        _emit_json(
            [
                {
                    "path": path,
                    "redaction_count": result.redaction_count,
                    "entity_counts": result.entity_counts,
                    **({} if check else {"text": result.text}),
                }
                for path, result in sources
            ]
        )
    elif check:
        for path, result in sources:
            if result.redaction_count:
                print(f"{path}: {result.redaction_count} PHI entities {result.entity_counts}")
    else:
        for _, result in sources:
            sys.stdout.write(result.text or "")
    if check and found:
        sys.exit(1)


def _cmd_guard_hook() -> None:
    """Pre-tool hook: exit 2 (block) when the targeted file contains PHI."""
    from .guard import check_hook_payload

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"auscult guard-hook: invalid hook payload: {exc}", file=sys.stderr)
        sys.exit(2)
    decision = check_hook_payload(payload if isinstance(payload, dict) else {})
    if not decision.allow:
        print(f"Blocked by auscult guard-hook: {decision.reason}", file=sys.stderr)
        sys.exit(2)


def _cmd_codex(
    prompt: str | None,
    *,
    events: str | None,
    agent_type: str,
    codex_args: list[str],
    as_json: bool,
) -> None:
    from .integrations.codex import ingest_codex_events, run_codex

    if events is not None:
        result = ingest_codex_events(events, prompt=prompt, agent_type=agent_type)
    elif prompt is None:
        print("auscult codex: a prompt is required unless --events is given", file=sys.stderr)
        sys.exit(2)
    else:
        try:
            result = run_codex(prompt, agent_type=agent_type, codex_args=codex_args)
        except FileNotFoundError:
            print("auscult codex: the `codex` CLI was not found on PATH", file=sys.stderr)
            sys.exit(2)

    if as_json:
        _emit_json(asdict(result))
    else:
        if result.final_message:
            print(result.final_message.rstrip())
            print()
        status = "failed" if result.failed else "completed"
        print(f"Run ID: {result.run_id} ({status})")
        if result.thread_id:
            print(f"Codex thread: {result.thread_id}")
    if result.failed:
        sys.exit(result.exit_code or 1)


def _cmd_demo(*, as_json: bool) -> None:
    from .demo import main as demo_main
    from .demo import run_demo

    if as_json:
        _emit_json({"run_id": run_demo(), "database_url": os.environ["DATABASE_URL"]})
    else:
        demo_main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="auscult")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    runs_parser = subparsers.add_parser("runs", help="List runs in the database.")
    runs_parser.add_argument("--agent-type", help="Filter by agent_type.")
    runs_parser.add_argument("--status", help="Filter by status (e.g. completed, failed).")
    runs_parser.add_argument(
        "--since",
        type=_parse_since,
        help="Only runs started at/after this ISO-8601 timestamp.",
    )
    runs_parser.add_argument("--limit", type=int, help="Maximum number of runs to return.")

    run_parser = subparsers.add_parser("run", help="Show one run with its steps.")
    run_parser.add_argument("run_id", help="Run id to display.")

    steps_parser = subparsers.add_parser("steps", help="List the steps of a run.")
    steps_parser.add_argument("run_id", help="Run id whose steps to display.")

    replay_parser = subparsers.add_parser(
        "replay", help="Playback a sanitized run step by step."
    )
    replay_parser.add_argument("run_id", help="Run id to replay.")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Re-run a handler against a recorded run and diff outputs.",
    )
    compare_parser.add_argument("run_id", help="Run id to compare.")
    compare_parser.add_argument(
        "--handler",
        required=True,
        help="Callable in module:function form; signature (cmd) -> (output, error).",
    )

    stats_parser = subparsers.add_parser(
        "stats",
        help="Aggregate failure rate / steps / latency / redactions per agent_type.",
    )
    stats_parser.add_argument("--agent-type", help="Limit stats to one agent_type.")

    export_parser = subparsers.add_parser(
        "export",
        help="Export a sanitized run as JSON or JSONL (for offline QA / sharing).",
    )
    export_parser.add_argument("run_id", help="Run id to export.")
    export_parser.add_argument(
        "--format",
        dest="fmt",
        choices=("json", "jsonl"),
        default="json",
        help="Output format (default: json).",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        help="Write to this file instead of stdout.",
    )

    purge_parser = subparsers.add_parser(
        "purge",
        help="Delete finished runs started before a cutoff (retention).",
    )
    purge_parser.add_argument(
        "--before",
        type=_parse_since,
        required=True,
        help="Delete runs started before this ISO-8601 timestamp.",
    )
    purge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching runs without deleting.",
    )
    purge_parser.add_argument("--agent-type", help="Limit purge to one agent_type.")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Score PHI detection precision/recall on the golden corpus.",
    )
    eval_parser.add_argument(
        "--corpus",
        help="Path to a JSONL corpus (default: packaged phi_eval_corpus.jsonl).",
    )
    eval_parser.add_argument(
        "--nlp-model",
        help="Primary spaCy model (default: AUSCULT_SPACY_MODEL / en_core_web_lg).",
    )
    eval_parser.add_argument(
        "--dual-pass-model",
        help="Optional second spaCy model for PERSON/LOCATION ensemble recall.",
    )
    eval_parser.add_argument(
        "--score-threshold",
        type=float,
        help="Presidio score threshold override.",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="List examples with false positives/negatives.",
    )

    setup_parser = subparsers.add_parser(
        "setup",
        help="Download the spaCy NLP model used for PHI detection.",
    )
    setup_parser.add_argument(
        "--model",
        default=DEFAULT_SPACY_MODEL,
        help=f"spaCy model to download (default: {DEFAULT_SPACY_MODEL}).",
    )

    subparsers.add_parser(
        "migrate",
        help="Apply database migrations (creates runs/steps tables).",
    )

    subparsers.add_parser(
        "demo",
        help="Capture one run with synthetic data (defaults to a SQLite file in /tmp).",
    )

    codex_parser = subparsers.add_parser(
        "codex",
        help="Run `codex exec --json` (or ingest its saved output) as a captured run.",
    )
    codex_parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt for Codex (optional with --events, where it labels the run).",
    )
    codex_parser.add_argument(
        "--events",
        help="Ingest a saved `codex exec --json` JSONL file instead of running Codex.",
    )
    codex_parser.add_argument(
        "--agent-type", default="codex", help="agent_type for the run (default: codex)."
    )
    codex_parser.add_argument(
        "--codex-arg",
        dest="codex_args",
        action="append",
        default=[],
        help="Extra argument for `codex exec`; repeat as needed "
        "(e.g. --codex-arg=-s --codex-arg=read-only).",
    )

    sanitize_parser = subparsers.add_parser(
        "sanitize",
        help="Print files (or stdin) with PHI replaced, for agents to read.",
    )
    sanitize_parser.add_argument("paths", nargs="*", help="Files to sanitize (default: stdin).")
    sanitize_parser.add_argument(
        "--check",
        action="store_true",
        help="Report PHI counts instead of text; exit 1 if any PHI is found.",
    )

    subparsers.add_parser(
        "guard-hook",
        help="Pre-tool hook for coding agents: block reads of files that contain PHI.",
    )

    args = parser.parse_args()
    as_json = bool(args.json)

    # Commands that do not need a database.
    if args.command == "eval":
        _cmd_eval(
            corpus=args.corpus,
            nlp_model=args.nlp_model,
            dual_pass_model=args.dual_pass_model,
            score_threshold=args.score_threshold,
            verbose=args.verbose,
            as_json=as_json,
        )
        return
    if args.command == "setup":
        _cmd_setup(model=args.model, as_json=as_json)
        return
    if args.command == "sanitize":
        _cmd_sanitize(args.paths, check=args.check, as_json=as_json)
        return
    if args.command == "guard-hook":
        _cmd_guard_hook()
        return
    if args.command == "demo":
        os.environ.setdefault("DATABASE_URL", DEFAULT_DEMO_DATABASE_URL)
        _cmd_demo(as_json=as_json)
        return

    if "DATABASE_URL" not in os.environ:
        print(
            "DATABASE_URL is not set; cannot connect to the database.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.command == "migrate":
        _cmd_migrate(as_json=as_json)
    elif args.command == "runs":
        _cmd_runs(
            agent_type=args.agent_type,
            status=args.status,
            since=args.since,
            limit=args.limit,
            as_json=as_json,
        )
    elif args.command == "run":
        _cmd_run(args.run_id, as_json=as_json)
    elif args.command == "steps":
        _cmd_steps(args.run_id, as_json=as_json)
    elif args.command == "replay":
        _cmd_replay(args.run_id, as_json=as_json)
    elif args.command == "compare":
        _cmd_compare(args.run_id, args.handler, as_json=as_json)
    elif args.command == "stats":
        _cmd_stats(agent_type=args.agent_type, as_json=as_json)
    elif args.command == "export":
        _cmd_export(args.run_id, fmt=args.fmt, output=args.output)
    elif args.command == "codex":
        _cmd_codex(
            args.prompt,
            events=args.events,
            agent_type=args.agent_type,
            codex_args=args.codex_args,
            as_json=as_json,
        )
    elif args.command == "purge":
        _cmd_purge(
            before=args.before,
            dry_run=args.dry_run,
            agent_type=args.agent_type,
            as_json=as_json,
        )


if __name__ == "__main__":
    main()
