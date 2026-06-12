import argparse
import os
import sys

from sqlalchemy import select

from .db import get_session
from .models import Run, Step
from .replay import RunNotFoundError, RunReplayer, format_playback


def _print_run(run: Run) -> None:
    finished = run.finished_at.isoformat() if run.finished_at else "—"
    failed_step = run.failed_step if run.failed_step is not None else "—"
    print(f"run {run.id}")
    print(f"  agent_type:     {run.agent_type}")
    print(f"  status:         {run.status}")
    print(f"  total_steps:    {run.total_steps}")
    print(f"  failed_step:    {failed_step}")
    print(f"  started_at:     {run.started_at.isoformat()}")
    print(f"  finished_at:    {finished}")
    print(f"  initial_prompt: {run.initial_prompt}")


def _print_step(step: Step) -> None:
    print(f"step {step.step_index} ({step.id})")
    print(f"  llm_command: {step.llm_command}")
    print(f"  output:      {step.output if step.output is not None else '—'}")
    if step.error_message is not None:
        print(f"  error:       {step.error_message}")
    if step.time_for_completion is not None:
        print(f"  duration:    {step.time_for_completion:.3f}s")


def _get_steps(session, run_id: str) -> list[Step]:
    return (
        session.execute(
            select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
        )
        .scalars()
        .all()
    )


def _cmd_runs() -> None:
    with get_session() as session:
        runs = session.execute(select(Run).order_by(Run.started_at)).scalars().all()

        if not runs:
            print("No runs found.")
            return

        for run in runs:
            _print_run(run)
            print()


def _cmd_run(run_id: str) -> None:
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            print(f"Run {run_id} not found.", file=sys.stderr)
            sys.exit(1)

        _print_run(run)
        steps = _get_steps(session, run_id)

    print(f"\n{len(steps)} step(s):")
    for step in steps:
        print()
        _print_step(step)


def _cmd_replay(run_id: str) -> None:
    try:
        run = RunReplayer.from_run_id(run_id).run
    except RunNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(format_playback(run), end="")


def _cmd_steps(run_id: str) -> None:
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            print(f"Run {run_id} not found.", file=sys.stderr)
            sys.exit(1)

        steps = _get_steps(session, run_id)

    if not steps:
        print(f"No steps recorded for run {run_id}.")
        return

    for step in steps:
        _print_step(step)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="auscult")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("runs", help="List all runs in the database.")

    run_parser = subparsers.add_parser("run", help="Show one run with its steps.")
    run_parser.add_argument("run_id", help="Run id to display.")

    steps_parser = subparsers.add_parser("steps", help="List the steps of a run.")
    steps_parser.add_argument("run_id", help="Run id whose steps to display.")

    replay_parser = subparsers.add_parser(
        "replay", help="Playback a sanitized run step by step."
    )
    replay_parser.add_argument("run_id", help="Run id to replay.")

    args = parser.parse_args()

    if "DATABASE_URL" not in os.environ:
        print("DATABASE_URL is not set; cannot connect to the database.", file=sys.stderr)
        sys.exit(2)

    if args.command == "runs":
        _cmd_runs()
    elif args.command == "run":
        _cmd_run(args.run_id)
    elif args.command == "steps":
        _cmd_steps(args.run_id)
    elif args.command == "replay":
        _cmd_replay(args.run_id)


if __name__ == "__main__":
    main()
