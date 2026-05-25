import argparse

from sqlalchemy import select

from .db import SessionLocal
from .models import Run


def _cmd_runs() -> None:
    with SessionLocal() as session:
        runs = session.execute(select(Run).order_by(Run.started_at)).scalars().all()

    if not runs:
        print("No runs found.")
        return

    for run in runs:
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
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="auscult")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("runs", help="List all runs in the database.")

    args = parser.parse_args()

    if args.command == "runs":
        _cmd_runs()


if __name__ == "__main__":
    main()
