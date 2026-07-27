"""Command-line interface for inspecting and replaying sanitized runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import select

from .db import get_session
from .eval import evaluate, format_report
from .export import export_run, purge_runs_before
from .migrate import upgrade_head
from .models import Run, Step
from .replay import RunNotFoundError, RunReplayer, format_playback
from .sanitizer import DEFAULT_SPACY_MODEL, Sanitizer


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


def _run_to_dict(run: Run) -> dict[str, Any]:
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


def _step_to_dict(step: Step) -> dict[str, Any]:
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


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _get_steps(session: Any, run_id: str) -> list[Step]:
    return list(
        session.execute(
            select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
        )
        .scalars()
        .all()
    )


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

        runs = session.execute(query).scalars().all()

        if as_json:
            _emit_json([_run_to_dict(run) for run in runs])
            return

        if not runs:
            print("No runs found.")
            return

        for run in runs:
            _print_run(run)
            print()


def _cmd_run(run_id: str, *, as_json: bool) -> None:
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            print(f"Run {run_id} not found.", file=sys.stderr)
            sys.exit(1)

        steps = _get_steps(session, run_id)
        if as_json:
            payload = _run_to_dict(run)
            payload["steps"] = [_step_to_dict(step) for step in steps]
            _emit_json(payload)
            return

        _print_run(run)

    print(f"\n{len(steps)} step(s):")
    for step in steps:
        print()
        _print_step(step)


def _cmd_steps(run_id: str, *, as_json: bool) -> None:
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            print(f"Run {run_id} not found.", file=sys.stderr)
            sys.exit(1)

        steps = _get_steps(session, run_id)

    if as_json:
        _emit_json([_step_to_dict(step) for step in steps])
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
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if as_json:
        _emit_json(
            {
                "run_id": run.run_id,
                "agent_type": run.agent_type,
                "status": run.status,
                "failed_step": run.failed_step,
                "initial_prompt": run.initial_prompt,
                "steps": [
                    {
                        "step_index": step.step_index,
                        "llm_command": step.llm_command,
                        "output": step.output,
                        "error_message": step.error_message,
                        "time_for_completion": step.time_for_completion,
                    }
                    for step in run.steps
                ],
            }
        )
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
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if as_json:
        _emit_json(
            {
                "run_id": result.run_id,
                "all_matched": result.all_matched,
                "steps": [
                    {
                        "step_index": step.step_index,
                        "llm_command": step.llm_command,
                        "expected_output": step.expected_output,
                        "actual_output": step.actual_output,
                        "expected_error": step.expected_error,
                        "actual_error": step.actual_error,
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
        runs = session.execute(runs_query).scalars().all()

        by_type: dict[str, dict[str, Any]] = {}
        for run in runs:
            bucket = by_type.setdefault(
                run.agent_type,
                {
                    "agent_type": run.agent_type,
                    "runs": 0,
                    "failed_runs": 0,
                    "total_steps": 0,
                    "step_latencies": [],
                    "redactions": 0,
                },
            )
            bucket["runs"] += 1
            if run.status == "failed":
                bucket["failed_runs"] += 1
            bucket["total_steps"] += run.total_steps
            bucket["redactions"] += run.redaction_count

        run_ids = [run.id for run in runs]
        steps: list[Step] = []
        if run_ids:
            steps = list(
                session.execute(select(Step).where(Step.run_id.in_(run_ids)))
                .scalars()
                .all()
            )
        run_agent = {run.id: run.agent_type for run in runs}
        for step in steps:
            agent = run_agent.get(step.run_id)
            if agent is None or agent not in by_type:
                continue
            if step.time_for_completion is not None:
                by_type[agent]["step_latencies"].append(step.time_for_completion)

        rows = []
        for agent, bucket in sorted(by_type.items()):
            run_count = bucket["runs"]
            latencies: list[float] = bucket["step_latencies"]
            rows.append(
                {
                    "agent_type": agent,
                    "runs": run_count,
                    "failed_runs": bucket["failed_runs"],
                    "failure_rate": (
                        bucket["failed_runs"] / run_count if run_count else 0.0
                    ),
                    "avg_steps": (
                        bucket["total_steps"] / run_count if run_count else 0.0
                    ),
                    "avg_step_latency": (
                        sum(latencies) / len(latencies) if latencies else None
                    ),
                    "avg_redactions": (
                        bucket["redactions"] / run_count if run_count else 0.0
                    ),
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
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

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
    try:
        from spacy.cli.download import download as spacy_download
    except ImportError as exc:  # pragma: no cover - spacy is a hard dep
        raise SystemExit(f"spaCy is required to download models: {exc}") from exc

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
    elif args.command == "purge":
        _cmd_purge(
            before=args.before,
            dry_run=args.dry_run,
            agent_type=args.agent_type,
            as_json=as_json,
        )


if __name__ == "__main__":
    main()
