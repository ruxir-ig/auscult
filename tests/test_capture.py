from sqlalchemy import select

from auscult.capture import AuscultTracer
from auscult.db import get_session
from auscult.models import Run, Step

PHI_VALUES = [
    "John Smith",
    "212-555-0182",
    "john.smith@example.com",
    "48293012",
]


def _all_stored_text(run_id: str) -> str:
    with get_session() as session:
        run = session.get(Run, run_id)
        steps = (
            session.execute(select(Step).where(Step.run_id == run_id))
            .scalars()
            .all()
        )
        parts = [run.initial_prompt]
        for step in steps:
            parts.extend(
                p for p in [step.llm_command, step.output, step.error_message] if p
            )
    return "\n".join(parts)


def test_tracer_never_stores_raw_phi(db) -> None:
    tracer = AuscultTracer(
        agent_type="test-agent",
        initial_prompt="Triage John Smith, MRN: 48293012.",
    )
    tracer.record_step(
        llm_command="lookup_patient('John Smith')",
        output="Phone on file: 212-555-0182, email john.smith@example.com.",
    )
    tracer.record_step(
        llm_command="notify()",
        output=None,
        error_message="Could not reach John Smith at 212-555-0182.",
    )
    tracer.finish()

    stored = _all_stored_text(tracer.run_id)
    for phi in PHI_VALUES:
        assert phi not in stored


def test_tracer_marks_run_completed(db) -> None:
    tracer = AuscultTracer(agent_type="test-agent", initial_prompt="Check vitals.")
    tracer.record_step(llm_command="get_vitals()", output="BP 120/80.")
    tracer.finish()

    with get_session() as session:
        run = session.get(Run, tracer.run_id)
        assert run.status == "completed"
        assert run.total_steps == 1
        assert run.failed_step is None
        assert run.finished_at is not None


def test_tracer_marks_run_failed_at_first_error(db) -> None:
    tracer = AuscultTracer(agent_type="test-agent", initial_prompt="Check vitals.")
    tracer.record_step(llm_command="get_vitals()", output="BP 120/80.")
    tracer.record_step(llm_command="order_labs()", output=None, error_message="timeout")
    tracer.finish()

    with get_session() as session:
        run = session.get(Run, tracer.run_id)
        assert run.status == "failed"
        assert run.failed_step == 1
