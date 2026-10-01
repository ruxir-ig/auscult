"""Capture one run with synthetic example data (``auscult demo``)."""

from __future__ import annotations

import os

DEFAULT_DEMO_DATABASE_URL = "sqlite:////tmp/auscult-demo.sqlite"

# Synthetic data only: the name and date of birth are invented.
DEMO_PROMPT = "Patient Jane Example, born 04/12/1980, reports a rash."


def run_demo() -> str:
    """Apply migrations, capture one triage run, and return its id.

    Uses ``DATABASE_URL`` if set, else a SQLite file in ``/tmp``.
    """
    os.environ.setdefault("DATABASE_URL", DEFAULT_DEMO_DATABASE_URL)

    from .capture import AuscultTracer
    from .migrate import upgrade_head

    upgrade_head()

    with AuscultTracer(agent_type="triage-agent", initial_prompt=DEMO_PROMPT) as tracer:
        tracer.record_step(
            llm_command="Classify the patient's symptoms and recommend next steps.",
            output="The patient reports a rash. Recommend clinical review.",
        )
    return tracer.run_id


def display_database_url() -> str:
    """The active ``DATABASE_URL`` with any password masked."""
    from sqlalchemy.engine import make_url

    return make_url(os.environ["DATABASE_URL"]).render_as_string(hide_password=True)


def follow_up_commands(run_id: str) -> list[str]:
    """Shell commands to inspect and replay the demo run.

    The URL is only spelled out (as a separate ``export``) for the default
    demo database, which has no credentials. A configured ``DATABASE_URL`` is
    left to the environment so its password is never printed. Keeping the
    export on its own line means each ``auscult`` command can be prefixed
    with ``uv run`` as-is.
    """
    commands = [f"auscult run {run_id}", f"auscult replay {run_id}"]
    if os.environ["DATABASE_URL"] == DEFAULT_DEMO_DATABASE_URL:
        commands.insert(0, f"export DATABASE_URL={DEFAULT_DEMO_DATABASE_URL}")
    return commands


def main() -> None:
    run_id = run_demo()
    print(f"Run ID: {run_id}")
    print("Inspect and replay it with:")
    for command in follow_up_commands(run_id):
        print(f"  {command}")
    print("(From a source checkout, run the auscult commands as `uv run auscult ...`.)")


if __name__ == "__main__":
    main()
