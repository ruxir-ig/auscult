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


def main() -> None:
    run_id = run_demo()
    database_url = os.environ["DATABASE_URL"]
    print(f"Run ID: {run_id}")
    print(f"Inspect: DATABASE_URL={database_url} uv run auscult run {run_id}")
    print(f"Replay:  DATABASE_URL={database_url} uv run auscult replay {run_id}")


if __name__ == "__main__":
    main()
