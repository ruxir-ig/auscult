"""End-to-end integration tests across capture, storage, sanitization, and replay."""

from sqlalchemy import select

from auscult.capture import AuscultTracer
from auscult.db import get_session
from auscult.models import Run, Step
from auscult.replay import RunReplayer

RAW_PHI = [
    "John Smith",
    "212-555-0182",
    "john.smith@example.com",
    "48293012",
]


def _record_smoke_run() -> str:
    tracer = AuscultTracer(
        agent_type="fake-agent",
        initial_prompt="Diagnose symptoms for John Smith, MRN: 48293012.",
    )
    tracer.record_step(
        llm_command="ask_patient(symptoms)",
        output="John Smith reports headache and fever. Callback: 212-555-0182.",
    )
    tracer.record_step(
        llm_command="lookup_conditions(['headache', 'fever'])",
        output="Possible: flu, migraine, sinus infection.",
    )
    tracer.record_step(
        llm_command="recommend_next_step()",
        output="Order CBC panel and email results to john.smith@example.com.",
    )
    tracer.finish()
    return tracer.run_id


def test_end_to_end_capture_sanitize_and_store(migrated_db) -> None:
    run_id = _record_smoke_run()

    with get_session() as session:
        run = session.get(Run, run_id)
        assert run is not None
        assert run.agent_type == "fake-agent"
        assert run.status == "completed"
        assert run.total_steps == 3
        assert run.failed_step is None
        assert run.started_at is not None
        assert run.finished_at is not None

        steps = (
            session.execute(
                select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
            )
            .scalars()
            .all()
        )
        assert len(steps) == 3
        assert steps[0].llm_command == "ask_patient(symptoms)"
        assert steps[0].error_message is None
        assert steps[0].output is not None
        assert steps[0].time_for_completion is not None
        assert steps[2].llm_command == "recommend_next_step()"

        stored_text = "\n".join(
            [run.initial_prompt]
            + [
                part
                for step in steps
                for part in [step.llm_command, step.output, step.error_message]
                if part
            ]
        )
        for phi in RAW_PHI:
            assert phi not in stored_text

        assert "headache" in steps[0].output
        assert "flu" in steps[1].output


def test_end_to_end_replay_loads_recorded_run(migrated_db) -> None:
    run_id = _record_smoke_run()

    replayer = RunReplayer.from_run_id(run_id)
    steps = list(replayer.playback())

    assert replayer.run.run_id == run_id
    assert replayer.run.status == "completed"
    assert len(steps) == 3
    assert steps[0].llm_command == "ask_patient(symptoms)"
    assert steps[0].output is not None
    assert "John Smith" not in (steps[0].output or "")
